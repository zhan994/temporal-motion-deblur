import os
from pprint import pformat
import torch
from mmengine.runner import set_random_seed
from util.data_util import save_sample, read_video_from_path
from util.registry import MODELS, build_module
from util.config_utils import parse_configs
from util.inference_utils import prepare_model_args
from util.misc import create_logger, to_torch_dtype

from model.vqgan.video_vqgan import VideoVQGANNoQuant
from model.measurements import get_noise, get_operator
from model.condition_methods import get_conditioning_method
from model.iddpm import IDDPM


def main():
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == seed ==
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == vqgan config ==
    vqgan = VideoVQGANNoQuant(cfg.vq_config, cfg.vq_ckpt, cfg.latent_std)
    vqgan = vqgan.to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", (64, 64))
    num_frames = cfg.get("num_frames", 10)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vqgan.get_latent_size(input_size)

    model = (
        build_module(cfg.model, MODELS, input_size=latent_size, in_channels=vqgan.out_channels).to(device, dtype).eval()
    )

    # == diffusion sampling scheduler ==
    scheduler = IDDPM(device, cfg.num_sampling_steps)

    # ======================================================
    # inference
    # ======================================================
    # == dps prepare ==
    operator = get_operator(name='vid_motion_blur', vqgan=vqgan, device=device, dtype=dtype)
    noiser = get_noise(name='gaussian', sigma=cfg.noise_level)
    cond_method = get_conditioning_method('ps', operator, noiser, scale=cfg.dps_scale)
    measurement_cond_fn = cond_method.conditioning

    # == reference ==
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    ref_vid = read_video_from_path(cfg.ref_path, image_size=image_size)
    save_sample(ref_vid, fps=cfg.fps, save_path=os.path.join(save_dir, f"reference"))

    # == blurry img / ob ==
    ob = operator.forward(vqgan.encode(ref_vid.unsqueeze(0).to(device, dtype)))
    ob_n = noiser(ob)
    save_sample(ob_n.squeeze(0), fps=cfg.fps, save_path=os.path.join(save_dir, f"sample_ob"))


    sample_idx = 0
    for i in range(0, 10):
        model_args = prepare_model_args(1, image_size, num_frames, cfg.fps, device, dtype)
        z = torch.randn(1, vqgan.out_channels, *latent_size, device=device, dtype=dtype)
        samples = scheduler.sample(
            model,
            z=z,
            device=device,
            additional_args=model_args,
            ob=ob_n,
            measurement_cond_fn=measurement_cond_fn
        )
        samples = vqgan.decode(samples.to(dtype))
        samples = samples.to(dtype)

        print("sample_index", sample_idx)
        save_path = os.path.join(cfg.save_dir,f"sample_{sample_idx}")
        save_sample(samples.squeeze(0), fps=cfg.fps, save_path=save_path)
        sample_idx += 1


if __name__ == "__main__":
    main()