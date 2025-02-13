num_frames = 10
frame_interval = 1
fps = 2
image_size = (64, 64)

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained="PRETRAINED_MODEL",
    input_sq_size=64,
    qk_norm=False,
    qk_norm_legacy=False,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
)


num_sampling_steps = 1000
dtype = "bf16"
seed = 42
ref_path = "data/test_data/bair/main/00000634.mp4"
save_dir = "./outputs/bair_main/"

# dps params
dps_scale = 1.5
noise_level = 0.01

# vqgan params
vq_config = "configs/vqgan/vqgan_no_quant.yaml"
vq_ckpt = "pretrained/bair/vqgan.ckpt"
latent_std = 1.118574
