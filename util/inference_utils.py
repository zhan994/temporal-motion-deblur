import torch


def prepare_model_args(batch_size, image_size, num_frames, fps, device, dtype):
    fps = torch.tensor([fps], device=device, dtype=dtype).repeat(batch_size)
    height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(batch_size)
    width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(batch_size)
    num_frames = torch.tensor([num_frames], device=device, dtype=dtype).repeat(batch_size)
    ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(batch_size)
    return dict(height=height, width=width, num_frames=num_frames, ar=ar, fps=fps)


