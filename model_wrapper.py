import torch
import copy
import clip
import numpy as np
from transformers import DistilBertModel, DistilBertConfig
from transformers import AlbertModel, AlbertConfig
from transformers import RobertaModel,RobertaConfig
from torchvision.models import resnet50
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class EXIFNet(torch.nn.Module):
    def __init__(self, model, avg_word_embs=False, transformer="BERT", visual="RN50", resolution=124, pretrained=True):
        super().__init__()
        self.model = model
        del self.model.visual
        del self.model.transformer
        if visual=="RN50": image_encoder = resnet50(pretrained=True)
        else: raise NotImplementedError
        image_encoder.fc = torch.nn.Linear(2048, 768)
        self.model.visual = image_encoder
        
        if transformer == 'BERT':
            if pretrained: self.model.transformer = DistilBertModel.from_pretrained("distilbert-base-uncased")
            else: 
                configuration = DistilBertConfig()
                self.model.transformer = DistilBertModel(configuration)
        elif transformer == "ALBERT":
            if pretrained: self.model.transformer = AlbertModel.from_pretrained("albert-base-v2")
            else:
                configuration = AlbertConfig()
                self.model.transformer = AlbertModel(configuration)
        elif transformer == "ROBERTA":
            if pretrained: self.model.transformer = RobertaModel.from_pretrained("roberta-base")
            else: 
                configuration = RobertaConfig()
                self.model.transformer = RobertaModel(configuration)
        else: raise NotImplementedError
        
        
        self.avg_word_embs = avg_word_embs
        self.sink_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_text(self, inputs):
        if self.avg_word_embs:
            sequence_output = self.model.transformer(**inputs)[0]
            embeddings = torch.sum(
                sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

            return embeddings
        else:
            return self.model.transformer(**inputs)[1]
        
    def encode_image(self, image):
        return self.model.visual(image)
    
    def forward(self, image, attention_mask, input_ids):
        image, attention_mask, input_ids = image.to(DEVICE), attention_mask.to(DEVICE), input_ids.to(DEVICE)
        text = {'attention_mask': attention_mask, 'input_ids': input_ids}
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(text)
        return image_embeds, text_embeds
    
    

        

        
    
        
def load_wrapper_model(device, state_dict_path=None, split_gpus=False, model_name="ViT-B/32", input_resolution=124, transformer="BERT", visual="RN50", pretrained=True):
    
    clipNet, _ = clip.load(model_name,device=device,jit=False)
    if device == "cpu":
          clipNet.float()
    else :
        clip.model.convert_weights(clipNet)
    logit_scale = clipNet.logit_scale.exp()
    
    clipNet = EXIFNet(clipNet, avg_word_embs=True, transformer=transformer, visual=visual, resolution=input_resolution, pretrained=pretrained)
    
    
    if state_dict_path: 
        if device == 'cpu': checkpoint = torch.load(state_dict_path, map_location=torch.device('cpu'))
        else: checkpoint = torch.load(state_dict_path)
        clipNet.load_state_dict(checkpoint['model'])
        print(f"load pretraining model {state_dict_path}")
        
    if split_gpus:
        print("Multiple GPU setting!")
        if torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), "GPUs!")
        clipNet = torch.nn.DataParallel(clipNet)
    clipNet.to(device)
    
    return clipNet, logit_scale.detach()

