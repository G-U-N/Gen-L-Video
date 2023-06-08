import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import sys

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
from glv.models.unet import UNet3DConditionModel, Adapter
from glv.data.dataset import GLVDataset
from glv.pipelines.pipeline_one_shot_tuning import OneShotTuningPipeline
from glv.util import ddim_inversion_long, save_videos_grid, ddim_inversion
from einops import rearrange
import copy
import numpy as np
from glv.lora_util import inject_trainable_lora,get_lora

from glv.models_wota.controlnet import ControlNetModel
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    start_validatoin_steps: int = 300,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    cond_prob = 0.7,
    clip_drop_prob = 0.3,
    lora_r = 16,
    adapter_path: str = None,
    controlnet_path = None,
    controlnet_scale = 1.0,
    run_isolated = False,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    
    

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
        os.makedirs(f"{output_dir}/multiinv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    logfilename = os.path.join(output_dir,"exp")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    
    if adapter_path is not None:
        adapter = Adapter(
            cin=64 * 3 if ("sketch" not in adapter_path and "canny" not in adapter_path) else 64*1,
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False)
        adapter.load_state_dict(torch.load(adapter_path))
    else:
        adapter = None
    unet.adapter = adapter
    
    if controlnet_path is not None:
        controlnet = ControlNetModel.from_pretrained_2d(controlnet_path)
    else:
        controlnet = None
    unet.controlnet = controlnet
    unet.controlnet_scale = controlnet_scale
    
    # inject_trainable_lora(unet,["CrossAttentionWithLora"],r=lora_r,stride=validation_data.stride)
    
    ns, ms, m_parents = [], [], []
    for n, m in unet.named_modules():
        if ("attn1" in n) and "to_q" in n:
            print("inssert",n)
            *path, name = n.split(".")
            m_parent = unet
            while path:
                m_parent = m_parent.get_submodule(path.pop(0))
            m_parent._modules[name] = get_lora(m,r=lora_r,stride=validation_data.stride)
            ns.append(name)
            ms.append(m)
            m_parents.append(m_parent)
    for m_parent, m, name in zip(m_parents,ms,ns):
        m_parent._modules[name] = get_lora(m,r=lora_r, stride=validation_data.stride,num_loras=50)
    

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_data["stride"] = validation_data["stride"] 
    train_dataset = GLVDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]
    
    train_dataset.null_prompt_ids = tokenizer(
      "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, 
    )

    # Get the validation pipeline
    validation_pipeline = OneShotTuningPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    if adapter is not None:
        adapter.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    train_loss_avg = 0.0
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            # max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
            # print(f"max memory allocated: {max_memory_allocated:.3f} GB.")
            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
                
                clip_id = batch["clip_id"]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                mask = torch.from_numpy(np.random.choice([1.0,0.0],size=bsz,p=[cond_prob, 1-cond_prob])).to(latents.device,weight_dtype)
                mask = mask.unsqueeze(1).unsqueeze(2)
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0] * mask + text_encoder(batch["null_prompt_ids"])[0] * (1-mask)
                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
                for step, prob in clip_drop_prob:
                    if global_step >= step:
                        continue
                    mask = torch.from_numpy(np.random.choice([1,0],size=bsz,p=[prob, 1-prob])).to(latents.device).long()
                    break
                if sum(mask == 0) == 0:
                    clip_id = None
                else:
                    clip_id = -torch.ones_like(clip_id)*mask + clip_id * (1-mask)
                    
                control = batch.get("control_video")
                if control is not None:
                    control= rearrange(control, "b f c h w -> b c f h w")
                model_pred = unet(noisy_latents, timesteps, clip_id, encoder_hidden_states,control=control).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss_avg += ((train_loss-train_loss_avg)/global_step)
                logging.info(f"global_step: {global_step}, train loss avg:  {train_loss_avg:.5f}, train loss : {train_loss:.5f}")
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logging.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0 and global_step > start_validatoin_steps:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)
                        full_video = batch["full_video"][0].unsqueeze(0).to(weight_dtype)
                        full_control_video = batch.get("full_control_video")
                        if full_control_video is not None:
                            full_control_video = rearrange(full_control_video, "b f c h w -> b c f h w")
                            full_control_video = full_control_video[0].unsqueeze(0).to(weight_dtype)
                        video_length = full_video.shape[1]
                        full_video = rearrange(full_video, "b f c h w -> (b f) c h w")
                        clip_length = validation_data.video_length
                        ddim_inv_latent = None
                        if run_isolated:
                            if validation_data.use_inv_latent:
                                ddim_inv_latent_lst = []
                                for i in range(0,video_length-clip_length+1,clip_length):
                                    control = full_control_video[:,:,i:i+clip_length] if full_control_video is not None else None 
                                    latents = vae.encode(full_video[i:i+clip_length]).latent_dist.sample()
                                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=clip_length)
                                    latents = latents * 0.18215
                                    ddim_inv_latent = ddim_inversion(
                                    validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=validation_data.num_inv_steps, prompt="", clip_id = i, control=control)[-1].to(weight_dtype)
                                    ddim_inv_latent_lst.append(ddim_inv_latent)

                                inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                                ddim_inv_latent = torch.cat(ddim_inv_latent_lst,dim=2)
                                torch.save(ddim_inv_latent, inv_latents_path)

                            for idx, prompt in enumerate(validation_data.prompts):
                                sample_lst = []
                                for i in range(0,video_length-clip_length+1,clip_length):
                                    control = full_control_video[:,:,i:i+clip_length] if full_control_video is not None else None 
                                    sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent[:,:,i:i+clip_length], clip_id=i,control=control,
                                                             **validation_data).videos
                                    sample_lst.append(sample)
                                sample = torch.cat(sample_lst,dim=2)
                                save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                                samples.append(sample)
                            samples = torch.concat(samples)
                            save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                            save_videos_grid(samples, save_path)
                            logging.info(f"Saved samples to {save_path}")

                        if validation_data.use_inv_latent:
                            latents_lst = []
                            for i in range(0,video_length-clip_length+1,clip_length):
                                latents = vae.encode(full_video[i:i+clip_length]).latent_dist.sample()
                                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=clip_length)
                                latents = latents * 0.18215
                                latents_lst.append(latents)
                            latents = torch.cat(latents_lst,dim=2)
                                # print(latents.shape)
                            ddim_inv_latent = ddim_inversion_long(
                            validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                            num_inv_steps=validation_data.num_inv_steps, prompt="",window_size=clip_length,stride=validation_data.stride,control=full_control_video)[-1].to(weight_dtype)
                            inv_latents_path = os.path.join(output_dir, f"multiinv_latents/ddim_latent-{global_step}.pt")
                            torch.save(ddim_inv_latent, inv_latents_path)
                        samples = []
                        for idx, prompt in enumerate(validation_data.prompts):
                            validation_multidata = copy.deepcopy(validation_data)
                            validation_multidata.video_length = ddim_inv_latent.shape[2]
                            sample = validation_pipeline.gen_long(prompt, generator=generator, latents=ddim_inv_latent, window_size=clip_length,control=full_control_video,
                                                     **validation_multidata).videos
                            save_videos_grid(sample, f"{output_dir}/multisamples/sample-{global_step}/{prompt}.gif")
                            samples.append(sample)
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/multisamples/sample-{global_step}.gif"
                        save_videos_grid(samples, save_path)
                        logging.info(f"Saved samples to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = OneShotTuningPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


adapter_paths={
    "pose":"[your path]/t2iadapter_openpose_sd14v1.pth",
    "sketch":"[your path]/t2iadapter_sketch_sd14v1.pth",
    "seg": "[your path]/t2iadapter_seg_sd14v1.pth",
    "depth":"[your path]/t2iadapter_depth_sd14v1.pth",
    "canny":"[your path]/t2iadapter_canny_sd14v1.pth"
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/one-shot-tuning/hike.yaml")
    parser.add_argument("--control",type=str,default=None)
    args = parser.parse_args()
    
    conf = OmegaConf.load(args.config)
    control = args.control
    instance= conf["video_name"]
    video_name = conf["video_name"] + ".mp4"
    if control is None:  
        conf["output_dir"] = os.path.join(conf["output_dir"],instance)
    else:
        conf["output_dir"] = os.path.join("{}-{}".format(conf["output_dir"][:-1],control),instance)
    if control is None:
        conf["train_data"]["video_path"] = os.path.join("./data",video_name)
    else:
        conf["train_data"]["video_path"] = os.path.join("./t_data",video_name)
        conf["train_data"]["control_path"] = os.path.join(os.path.join("./c_data",control),video_name)
        adapter_path = adapter_paths[control]
        conf["adapter_path"] = adapter_path
        if control == "sketch" or control == "canny":
            conf["train_data"]["control_channels"] = 1
        else:
            conf["train_data"]["control_channels"] = 3
    del conf["video_name"]
    main(**conf)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"max memory allocated: {max_memory_allocated:.3f} GB.")