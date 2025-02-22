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
ref_path = "data/test_data/clev/low_fps/17118_2.mp4"
save_dir = "./outputs/cle_f4/"

# dps params
# low_fps
dps_scale = 0.8
# raw
# dps_scale = 1.3
noise_level = 0.01

# vqgan params
vq_config = "configs/vqgan/vqgan_no_quant.yaml"
vq_ckpt = "pretrained/clev/vqgan.ckpt"
latent_std = 0.21848887