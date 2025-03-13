# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from phi_data import generate_data
import torch.nn.functional as F
import math

def main(args, data=None):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    # latent_size = args.image_size // 8
    # latent_size = (args.image_height, args.image_width)
    model_height = 400
    model_width = 480
    latent_size = (model_height, model_width)
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    # del state_dict['pos_embed']
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # return 0

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # y = torch.tensor(class_labels, device=device)


    if data is None:
        data = generate_data(height = args.image_height, width = args.image_width)[1]
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    save_image(data, "sample_input.png", nrow=1, normalize=True, value_range=(-1, 1))
    C, H, W = data.shape
    assert C == 1
    data = data.squeeze(0)
    rows = math.ceil(H / model_height)
    cols = math.ceil(W / model_width)
    n = rows * cols

    patch_size = latent_size
    pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
    pad_data = F.pad(data, (0, pad_w, 0, pad_h))

    sub_data = []
    sub_rows, sub_cols = patch_size
    for i in range(0, H, sub_rows):
        for j in range(0, W, sub_cols):
            sub_matrix = pad_data[i:i+sub_rows, j:j+sub_cols]
            sub_data.append(sub_matrix.unsqueeze(0))
    stack_data = torch.stack(sub_data, dim=0).to(device)   #(n, 1, model_height, model_width)
    save_image(stack_data, "sample_stack_input.png", nrow=cols, normalize=True, value_range=(-1, 1))
    z = torch.randn(n, 1, model_height, model_width, device=device)


    # z = torch.randn(n, 1, model_height, model_width, device=device)
    # data = [generate_data(height = args.image_height, width = args.image_width) for _ in range(n)]

    # _y = [x[1] for x in data]
    # y = torch.stack(_y, dim=0).to(device)
    # save_image(y, "sample_input.png", nrow=1, normalize=True, value_range=(-1, 1))

    # _gt = [x[0] for x in data]
    # gt = torch.stack(_gt, dim=0).to(device)
    # save_image(gt, "sample_gt.png", nrow=1, normalize=True, value_range=(-1, 1))

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # model_kwargs = dict(y=y)
    model_kwargs = dict(y=stack_data)

    # # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    # Save and display images:
    save_image(samples, "sample_stack_output.png", nrow=cols, normalize=True, value_range=(-1, 1))

    assert samples.shape == stack_data.shape
    samples = samples.squeeze(1)
    output = torch.zeros_like(pad_data)
    idx = 0
    for i in range(0, H, sub_rows):
        for j in range(0, W, sub_cols):
            output[i:i+sub_rows, j:j+sub_cols] = samples[idx]
            idx += 1
    output = output[0:H, 0:W]
    assert output.shape == data.shape

    output_ = output.unsqueeze(0)
    save_image(output_, "sample_output.png", nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    # parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--image-height", type=int, default=500)
    parser.add_argument("--image-width", type=int, default=2500)
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    data = None
    main(args, data)
