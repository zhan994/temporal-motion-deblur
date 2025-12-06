# Image Motion Blur Removal in the Temporal Dimension with Video Diffusion Models

[[Website]](https://zhan994.github.io/temporal-motion-deblur/)
[[arXiv]](https://arxiv.org/abs/2501.12604)
[[Hugging Face]](https://huggingface.co/zhan994/temporal-motion-deblur)

![merged_aux1_635](images/merged_aux1_635.png)

![merged_main_666](images/merged_main_666.png)

## Installation

```bash
conda create -n motion_deblur python=3.9
conda activate motion_deblur

# cuda-11.8 + torch-2.2.2
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Inference

To run inference with our provided weights, first download weights into folder [pretrained](https://huggingface.co/zhan994/temporal-motion-deblur)  as shown from **Hugging Face**.

```
pretrained/
├── bair
│   ├── ema.pt
│   └── vqgan.ckpt
└── clev
    ├── ema.pt
    └── vqgan.ckpt
```

For **BAIR** datasets, 

```bash
python sample_motion_deblur.py configs/bair/deblur.py --ckpt-path pretrained/bair/ema.pt
```

For **CLEVRER** datasets,

```bash
python sample_motion_deblur.py configs/clev/deblur.py --ckpt-path pretrained/clev/ema.pt
```

## Acknowledgement

- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling): Diffusion Posterior Sampling for General Noisy Inverse Problems.
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All.
- [VQGAN](https://github.com/CompVis/taming-transformers): Taming Transformers for High-Resolution Image Synthesis.

## Citation

```
@INPROCEEDINGS{11084505,
  author={Pang, Wang and Zhan, Zhihao and Zhu, Xiang and Bai, Yechao},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Image Motion Blur Removal In The Temporal Dimension With Video Diffusion Models}, 
  year={2025},
  volume={},
  number={},
  pages={325-330},
  keywords={Deblurring;Visualization;Technological innovation;Dynamics;Estimation;Training data;Transformer cores;Diffusion models;Transformers;Kernel;Motion deblurring;video diffusion model;diffusion transformer},
  doi={10.1109/ICIP55913.2025.11084505}}
```
