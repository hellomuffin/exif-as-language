<br />
<p align="center">
  <h54align="center">This is the official repo for the paper "EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata".</h4>

  <p align="center">  
    <a href="https://arxiv.org/abs/2301.04647">Paper</a>
    ·
    <a href="https://hellomuffin.github.io/exif-as-language/">Project Page</a>
    ·
    <a href="https://drive.google.com/drive/folders/1V9g3I2SoQtjAUz71hZeMutqoGpUiPl3u?usp=sharing">Model and dataset resource</a>
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## Overview
![Alt text](Images/firstpage_v3.png?raw=true "Title")

In this paper, we learn visual representations that convey camera properties by creating a joint embedding between image patches and photo metadata. This model treats metadata as a language-like modality: it converts EXIF tags that compose the metadata into a long piece of text, and processes it using an off-the-shelf model from natural language processing. We demonstrate the effectiveness of our learned features on a variety of downstream tasks that require an understanding of low-level imaging properties, where it outperforms other feature representations. In particular, we successfully apply our model to the problem of detecting image splices "zero shot", by clustering the crossmodal embeddings within an image.


#### This repository contains
* install requirements for this repo
* dataset preparing tips & training the model
* Inference on image forensic task

## Requirements
After cloning our repo, please run

`pip install -r requirements.txt`

If you want to train models by your own, we recommand also install pytorch_warmup via `pip install -U pytorch_warmup`



## Training a new model
If you wish to train your own model,  you can follow the following things:

#### Prepare a set of image-exif pairs from some image-text dataset
EXIF information is a metadata file injected in the photo at the moment of capture. Some social media platform will remove the EXIF data in the user uploaded photos for privacy protection. Therefore, instead of extracting EXIF data directly from photos, we recommand using pubic datasets in which EXIF information for every image is provided, such as  [LAION](https://laion.ai/blog/laion-400-open-dataset/) and [YFCC](https://paperswithcode.com/dataset/yfcc100m) dataset.

For example, to download YFCC dataset, first get the dataset metadata from:

`s3cmd get --recursive s3://mmcommons `

Then download images based on metadata:

`python dataProcess/download_image.py --target_folder </path/to/target/folder> --metadata_folder </path/to/metadata/folder> --sample_size <integer>`

Finally downloading EXIF info 

`python yfcc_dataInfo.py --img_folder_path </path/to/image/folder> --metadata_path <path/to/exif/data/path>`



#### Write pytorch dataset code
After downloading the data, you will need to write a customized pytorch dataset code for training. We give an example of dataset class on `data.py`.

#### start training
Now you can formally start training. 


`python train.py --save_model_path your/checkpoint/path --batch_size <batch_size> --num_epochs <num_epoch>`


add `--multi_gpu` in the case of distributed gpus training. We recommand using [WandB](https://wandb.ai/site) to track the statistics while training, such as loss, gradient, etc. To use it, simple add `--logWandb`.


## Evaluating on image forensics task
We provide evaluation code on various forensics datasets, including CASIA, Columbia, DSO, In the wild, Realistic Tampering and scene completion. After downloading those datasets and put it on the path `your-eval-data-root-path`. After that, you can run

`python splice_evaluator.py --ckpt_path /your-pretrained-model-path --result_dir result --data_name in_the_wild --data_base_path /your-eval-data-root-path`

where data_name specify which dataset you want to run, and can be choose from `['in_the_wild', 'columbia', 'dso_1', 'realistic_tampering', 'scene_completion', 'casia_1', 'casia_2']`. After evaluating, the similarity heatmap for each image will be placed on `result/` and the evaluating score will be displayed via standard output.

## The released checkpoint performance

The pretraining dataset we use is a 1.5M subset sampled randomly from the YFCC100M dataset, but all the experimental setup is the same as the  [original paper](https://arxiv.org/pdf/2301.04647.pdf).

#### Linear probing experiment:

Forensics evaluation:

| Dataset                 | CASIA 1 (resized) | CASIA 1 (cropped) | CASIA 2 (resized) | CASIA 2 (resized) |
| ----------------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Linear probing accuracy | 0.75              | 0.84              | 0.87              | 0.84              |

Radial distortion parameter prediction

| Dataset                 | Dresden | RAISE |
| ----------------------- | ------- | ----- |
| Linear probing accuracy | 0.31    | 0.35  |

#### Zero-shot splice detection and localization

Zero-shot splice localization

| Dataset | Columbia | DSO  | RT   | In-the-Wild | Hays |
| ------- | -------- | ---- | ---- | ----------- | ---- |
| p-mAP   | 0.94     | 0.62 | 0.23 | 0.54        | 0.31 |

Zero-shot splice detection

| Dataset | Columbia | DSO  | RT   |
| ------- | -------- | ---- | ---- |
| mAP     | 0.98     | 0.66 | 0.53 |



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Fake detection lab](https://github.com/yizhe-ang/fake-detection-lab)
* [CLIP](https://openai.com/blog/clip/)

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- CITATION -->
## Citing
If you found this repository useful, please consider citing:

```bibtex
@inproceedings{zheng2023exif,
  title={EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata},
  author={Zheng, Chenhao and Shrivastava, Ayush and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6945--6956},
  year={2023}
}
```
