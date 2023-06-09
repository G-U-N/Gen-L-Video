
import argparse
import copy
import logging
import inspect
import os
from typing import Dict, Optional
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from glv.models_wota.unet import UNet3DConditionModel
from glv.models_wota.controlnet import ControlNetModel
from glv.data.dataset import GLVDataset
from glv.pipelines.pipeline_tuning_free_control import TuningFreeControlPipeline
from glv.util import ddim_inversion_long, save_videos_grid, ddim_inversion
from einops import rearrange

check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    train_batch_size: int = 1,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    controlnet_path = None,
    controlnet_scale = 1.0,

):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    if controlnet_path is not None:
        controlnet = ControlNetModel.from_pretrained_2d(controlnet_path)
    else:
        controlnet = None
    unet.controlnet = controlnet
    unet.controlnet_scale = controlnet_scale
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae.enable_slicing()
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if controlnet is not None:
        controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the training dataset
    train_dataset = GLVDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Get the validation pipeline
    validation_pipeline = TuningFreeControlPipeline(
        vae=vae, text_encoder=text_encoder, unet=unet, tokenizer=tokenizer, 
        scheduler=noise_scheduler
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)


    # Prepare everything with our `accelerator`.

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device,dtype=weight_dtype)
    if controlnet is not None:
        controlnet.to(accelerator.device, dtype=weight_dtype)
    # unet = accelerator.prepare(unet)
    if accelerator.is_main_process:
        accelerator.init_trackers("tuning-free t2v")

    if accelerator.is_main_process:
        for step, batch in enumerate(train_dataloader):
            logger.info("inference pixel values")
            pixel_values = batch["full_video"].to(accelerator.device,weight_dtype)[0].unsqueeze(0)
            video_length = pixel_values.shape[1]
            video_length = video_length - video_length % validation_data.video_length
            pixel_values = pixel_values[:,:video_length]
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            latents = [ ]
            for i in range(0,video_length,validation_data.video_length):
                latents.append( vae.encode(pixel_values[i:i+validation_data.video_length]).latent_dist.sample())
            latents = torch.cat(latents,dim=0)
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215
            pixel_values = pixel_values
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)
            clip_length = validation_data.video_length
            samples = []
            control = batch.get("full_control_video")
            if control is not None:
                control= rearrange(control, "b f c h w -> b c f h w")
                control = control[:,:,:video_length]
                control = control.to(accelerator.device,weight_dtype)
            for idx, prompt in enumerate(validation_data.prompts):
                with torch.autocast("cuda"):
                    validation_multidata = copy.deepcopy(validation_data)
                    validation_multidata.video_length = video_length
                    sample = validation_pipeline.gen_long(prompt,latents, generator=generator,window_size=validation_data.video_length,control=control,
                                             **validation_multidata).videos
                save_videos_grid(sample, f"{output_dir}/samples/sample/{idx}-{prompt[:32]}.gif")
                samples.append(sample)
            samples = torch.concat(samples)
            save_path = f"{output_dir}/samples/sample.gif"
            save_videos_grid(samples, save_path)
            logger.info(f"Saved samples to {save_path}")
            break
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuning-free-control/girl-glass.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"max memory allocated: {max_memory_allocated:.3f} GB.")