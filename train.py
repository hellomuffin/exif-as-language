import argparse
import os
import torch
from torch.utils.data import DataLoader
from pytorch_warmup import LinearWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DistilBertTokenizer
import wandb
from tqdm import tqdm

from data import ExampleDataset
from utils.general import create_logits, convert_models_to_fp32
from model_wrapper import load_wrapper_model

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"


class Trainer():
    def __init__(self, 
        model, 
        dataset_config, 
        training_config, 
        wandb_config,
        ckpt_config,
        logWandb=True, 
    ):
        self.model = model
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.ckpt_config = ckpt_config
        self.log = logWandb
        
        
        if logWandb:
            wandb.init(
            # Set the project where this run will be logged
                project = wandb_config["projectName"], 
                name = wandb_config["expName"], 
                config = {**dataset_config, **training_config}
            )
        
        
        self.loss_img_func = torch.nn.CrossEntropyLoss()
        self.loss_txt_func = torch.nn.CrossEntropyLoss()
        
        
        self.train_dataset = ExampleDataset(
            resolution=dataset_config["resolution"]
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size = self.training_config["batch_size"],
            shuffle=True, 
            num_workers=os.cpu_count()
        )
            
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.training_config["lr"], betas=(0.9,0.98), eps=1e-6, weight_decay=0.0001) 
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.training_config["num_epochs"], eta_min=1e-7)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=self.training_config["warm_up_epochs"]*len(self.train_dataloader))
    

    def __call__(self, multi_gpu=False):
        for epoch in range(self.training_config["num_epochs"]):
            loss_list = []
            for batch in tqdm(self.train_dataloader, desc="training...", total=len(self.train_dataloader)):
                loss = self.train_step(batch)
                loss_list.append(loss.item())
                if self.log:  wandb.log({"loss":loss})
            if self.log: wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})
            if epoch >= self.training_config["warm_up_epochs"]:
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            if multi_gpu: save_model = self.model.module
            else: save_model = self.model
            checkpoint = { 
                'epoch': epoch,
                'model': save_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_sched': self.lr_scheduler.state_dict()
            }
            torch.save(checkpoint,  self.ckpt_config["path"])
                
                    
    def train_step(self, batch):
        images, texts = batch["image"], batch["exif"]
        texts_tokenize = self.tokenizer(texts, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        image_embedding, text_embedding = self.model(images,texts_tokenize["attention_mask"], texts_tokenize['input_ids'])
        logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, self.training_config["logit_scale"])
        ground_truth = torch.arange(image_embedding.shape[0],dtype=torch.long,device=DEVICE)
        total_loss = (self.loss_img_func(logits_per_image,ground_truth) + self.loss_txt_func(logits_per_text,ground_truth))/2
        total_loss.backward()
        if DEVICE == "cpu": self.optimizer.step()
        else : 
            # convert_models_to_fp32(self.model)
            self.optimizer.step()
            # clip.model.convert_weights(self.model)
        self.optimizer.zero_grad()
        with self.warmup_scheduler.dampening(): pass
        return total_loss
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projName", default="exif-as-language", help="name your project")
    parser.add_argument("--expName", default="train", help="name your experiment")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--save_model_path", default="checkpoints/wrapper.pth")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument('--pretrainPath', default=None)
    parser.add_argument('--multi_gpu', default=False, action='store_true', help='Bool type')
    parser.add_argument('--logWandb', default=False, action='store_true', help='Bool type')
    
    args = parser.parse_args()
    
    
    exifNet,logit_scale = load_wrapper_model(
        device=DEVICE, 
        state_dict_path=args.pretrainPath, 
        split_gpus=args.multi_gpu,
        input_resolution=args.patch_size,
    )
    
    dataset_config = {
        "resolution": args.patch_size
    }
    
    training_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "logit_scale": logit_scale,
        "warm_up_epochs": args.warm_up_epoch,
        "num_epochs": args.num_epochs,
    }
    
    ckpt_config = {
        "path": args.save_model_path,
    }
    
    wandb_config = {
        "projectName": args.projName,
        "expName": args.expName,
    }
    
    trainer = Trainer(exifNet, dataset_config, training_config, wandb_config, ckpt_config, logWandb=args.logWandb)
    
    trainer(multi_gpu=args.multi_gpu)