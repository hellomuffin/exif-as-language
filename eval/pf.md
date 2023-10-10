### Performance of latest released checkpoints (Update Oct. 10)

link to the checkpoints: 

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

