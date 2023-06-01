import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange

from glv.pipelines.pipeline_tuning_free_inpaint import prepare_mask_and_masked_image


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=2, fps=10):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet, clip_id,control):
    noise_pred = unet(latents, t, clip_id=clip_id, encoder_hidden_states=context,control=control)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt,clip_id,control):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, clip_id,control)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="",clip_id = None,control=None):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, clip_id, control=control)
    return ddim_latents

@torch.no_grad()
def ddim_inversion_long(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", window_size=16, stride=8,control=None,pixel_values=None, mask = None):
    if mask is not None:
        assert pixel_values is not None
        mask, masked_image = prepare_mask_and_masked_image(pixel_values, mask)
        bz, num_channels,video_length,height,width = video_latent.shape 
        mask, masked_image_latents = pipeline.prepare_mask_latents(
            mask,
            masked_image,
            bz,
            height * pipeline.vae_scale_factor,
            width * pipeline.vae_scale_factor,
            video_latent.dtype,
            video_latent.device,
            None,
            False,
        )
        depth_map = rearrange(torch.cat([mask,masked_image_latents],dim=1),"(b f) c h w -> b c f h w",f=video_length)  
    elif pixel_values is not None and hasattr(pipeline,"prepare_depth_map"):
        video_length = video_latent.shape[2]
        depth_map = pipeline.prepare_depth_map(
            pixel_values,
            None,
            1,
            False,
            video_latent.dtype,
            video_latent.device,
        )
        depth_map = rearrange(depth_map,"(b f) c h w -> b c f h w",f=video_length)
    else:
        depth_map = None
    ddim_latents = ddim_loop_long(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, window_size, stride,control,depth_map)
    return ddim_latents

@torch.no_grad()
def ddim_loop_long(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, window_size, stride,control,depth_map):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    video_length = latent.shape[2]
    views = get_views(video_length,window_size=window_size,stride=stride)
    count = torch.zeros_like(latent)
    value = torch.zeros_like(latent)
    for i in tqdm(range(num_inv_steps)):
        count.zero_()
        value.zero_()
        for t_start, t_end in views:
            control_tmp = None if control is None else control[:,:,t_start:t_end]
            latent_view = latent[:,:,t_start:t_end]
            t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
            if depth_map is not None:
                latent_input = torch.cat([latent_view,depth_map[:,:,t_start:t_end]],dim=1)
            else:
                latent_input = latent_view
            noise_pred = get_noise_pred_single(latent_input, t, cond_embeddings, pipeline.unet, t_start,control_tmp)
            latent_view_denoised = next_step(noise_pred, t, latent_view, ddim_scheduler)
            value[:,:,t_start:t_end] += latent_view_denoised
            count[:,:,t_start:t_end] += 1
        latent = torch.where(count>0,value/count,value)
        all_latent.append(latent)
    return all_latent

def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start,t_end))
    return views