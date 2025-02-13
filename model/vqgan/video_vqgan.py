import yaml
import torch
from omegaconf import OmegaConf
from model.vqgan.vqgan_no_quant import VQModelNoQuant
from einops import rearrange
import torch.nn as nn

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, no_quantize=True):
  if no_quantize:
    model = VQModelNoQuant(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

class VideoVQGANNoQuant(nn.Module):
    def __init__(self, config_ptah, ckpt_path, latent_std, no_quantize=True):
        super().__init__()
        self.config = load_config(config_ptah)
        self.custom_vqgan = load_vqgan(config=self.config, ckpt_path=ckpt_path, no_quantize=no_quantize)
        self.normalizer = 1 / latent_std
        self.out_channels = self.config.model.params.embed_dim
        ch_mult_length = len(self.config.model.params.ddconfig.ch_mult)
        self.patch_size = (1, 2 ** (ch_mult_length - 1), 2 ** (ch_mult_length - 1))

    def encode(self, x):
        """
           编码
           x: (B, C, T, H, W)
        """

        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        encoded = self.custom_vqgan.encode(x).mul_(self.normalizer)
        encoded = rearrange(encoded, "(B T) C H W -> B C T H W", B=B)
        return encoded

    def decode(self, x):
        """
            解码
            x: (B, C, T, H, W)
        """
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        decoded = self.custom_vqgan.decode(x / self.normalizer)
        decoded = rearrange(decoded, "(B T) C H W -> B C T H W", B=B)
        return decoded

    def reconstruct(self, x):
        z = self.encode(x)
        rec = self.decode(z)
        return rec

    # def get_latent_size(self, input_size):
    #     """
    #       获取latent的特征大小 [T, H/8, W/8]
    #     """
    #     for i in range(3):
    #         assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
    #     input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
    #     return input_size

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size