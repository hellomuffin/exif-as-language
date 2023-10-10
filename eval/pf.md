### Performance of latest released checkpoints (Update Oct. 10)

link to the checkpoints: 

The pretraining dataset we use is a 1.5M subset sampled randomly from the YFCC100M dataset, but all the experimental setup is the same as the  [original paper](https://arxiv.org/pdf/2301.04647.pdf).

#### Linear probing experiment:

Forensics evaluation:

| Dataset           | Linear probing accuracy |
| :---------------- | ----------------------- |
| CASIA 1 (resized) | 0.75                    |
| CASIA 1 (cropped) | 0.84                    |
| CASIA 2 (resized) | 0.87                    |
| CASIA 2 (resized) | 0.84                    |

Radial distortion parameter prediction

| Dataset | Linear probing accuracy |
| :------ | ----------------------- |
| Dresden | 0.31                    |
| RAISE   | 0.35                    |

#### Zero-shot splice detection and localization

Zero-shot splice localization

| Dataset     | p-mAP | cIoU |
| ----------- | ----- | ---- |
| Columbia    | 0.94  | 0.96 |
| DSO         | 0.62  | 0.82 |
| RT          | 0.23  | 0.72 |
| In-the-Wild | 0.54  | 0.80 |
| Hays        | 0.31  | 0.61 |

Zero-shot splice detection

| Dataset  | mAP  |
| -------- | ---- |
| Columbia | 0.98 |
| DSO      | 0.66 |
| RT       | 0.53 |