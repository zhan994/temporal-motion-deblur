# Image Motion Blur Removal in the Temporal Dimension with Video Diffusion Models
Wang Pang<sup>1</sup>\*, Zhihao Zhan<sup>2</sup>\*, Xiang Zhu<sup>2</sup>\*, and Yechao Bai<sup>1#</sup>

(*Equal contribution, #Corresponding author)

<sup>1</sup>Nanjing University, <sup>2</sup>TopXGun Robotics

### [[Project Page](https://zhan994.github.io/temporal-motion-deblur/)] [[arXiv](https://arxiv.org/abs/2501.12604)]

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

To run inference with our provided weights, first download weights into folder `pretrained`  as shown.

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
@misc{pang2025imagemotionblurremoval,
      title={Image Motion Blur Removal in the Temporal Dimension with Video Diffusion Models}, 
      author={Wang Pang and Zhihao Zhan and Xiang Zhu and Yechao Bai},
      year={2025},
      eprint={2501.12604},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.12604},
}
```