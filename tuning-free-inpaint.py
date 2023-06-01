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
from glv.pipelines.pipeline_tuning_free_inpaint import TuningFreePipelineInpaint
from glv.util import ddim_inversion_long, save_videos_grid, ddim_inversion
from einops import rearrange
from tqdm import tqdm

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
import numpy as np
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.amg import remove_small_regions
from PIL import Image


def load_groundingdino_model(model_config_path, model_checkpoint_path,device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def prompt2mask(original_image, caption,grounding_model=None,sam_predictor=None, device="cuda",box_threshold=0.25, text_threshold=0.25, num_boxes=2):
    def image_transform_grounding(init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None)  # 3, h, w
        return init_image, image

    if isinstance(original_image,torch.Tensor):
        original_image = original_image.detach().cpu().permute(1,2,0).numpy()
        original_image = (original_image * 255).round().astype("uint8")
        original_image = Image.fromarray(original_image)
        original_image.save("instance.png")
    image_np = np.array(original_image, dtype=np.uint8)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    _, image_tensor = image_transform_grounding(original_image)
    boxes, logits, phrases = predict(grounding_model,
                                     image_tensor, caption, box_threshold, text_threshold, device=device)
    H, W = original_image.size[1], original_image.size[0]
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

    final_m = torch.zeros((image_np.shape[0], image_np.shape[1]))

    if boxes.size(0) > 0:
        sam_predictor.set_image(image_np)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )
        fine_masks = []
        for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
            fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])        
        masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(masks)

        num_obj = min(len(logits), num_boxes)
        for obj_ind in range(num_obj):
            m = masks[obj_ind][0]
            final_m += m
    
    final_m = (final_m > 0).to('cpu').numpy()
    return final_m[None,:,:]

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img * 255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    return full_img, res


def get_sam_control(image,mask_generator):
    image = image.detach().cpu().permute(1,2,0).numpy()
    image = (image * 255).round().astype("uint8")
    masks = mask_generator.generate(image)
    full_img, res = show_anns(masks)
    return full_img, res


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    mask_prompt: str,
    train_data: Dict,
    validation_data: Dict,
    train_batch_size: int = 1,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    controlnet_path = None,
    controlnet_scale = 1.0,
    sam_checkpoint = None,
    groundingdino_checkpoint = None,
    groundingdino_config_file = None,

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
    grounding_model = load_groundingdino_model(groundingdino_config_file, groundingdino_checkpoint,accelerator.device)
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
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
    grounding_model.requires_grad_(False)
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
    validation_pipeline = TuningFreePipelineInpaint(
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
    
    if controlnet is not None:
        controlnet.to(accelerator.device, dtype=weight_dtype)
    grounding_model.to(accelerator.device)
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(accelerator.device))
    sam.to(device=accelerator.device,dtype=weight_dtype)
    mask_generator = SamAutomaticMaskGenerator(sam)   
    unet = accelerator.prepare(unet)
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
            with torch.autocast("cuda"):
                masked_pixel_values = []
                for i in tqdm(range(video_length)):
                    masked_pixel_values.append(torch.from_numpy(np.array(prompt2mask((pixel_values[i]+1)/2., mask_prompt,grounding_model,sam_predictor,accelerator.device), dtype=np.float32)).to(accelerator.device,weight_dtype).unsqueeze(0))
                masked_pixel_values = torch.cat(masked_pixel_values)
                save_videos_grid(rearrange(masked_pixel_values.cpu(),"(b f) c h w -> b c f h w",b=1),"./mask.gif")
                control = None
                if controlnet is not None:
                    control = []
                    seg = []
                    for i in tqdm(range(video_length)):
                        seg_map, control_map = get_sam_control((pixel_values[i]+1)/2.,mask_generator)
                        control.append(torch.from_numpy(control_map).float().permute(2,0,1).to(accelerator.device,weight_dtype).unsqueeze(0))
                        seg.append(torch.from_numpy(seg_map).float().permute(2,0,1).to(accelerator.device,weight_dtype).unsqueeze(0))
                    control = torch.cat(control)
                    seg = torch.cat(seg)
                    control = rearrange(control,"(b f) c h w -> b c f h w",b=1)
                    save_videos_grid(control.detach().cpu(),"./controlmap.gif")
                    save_videos_grid(rearrange(seg.detach().cpu(),"(b f) c h w -> b c f h w",b=1),"./segmap.gif")
            del sam_predictor
            del grounding_model
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
            print(f"max memory allocated: {max_memory_allocated:.3f} GB.")
            samples = []
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)
            ddim_inv_latent = None
            clip_length = validation_data.video_length
            if validation_data.use_inv_latent:
                inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent.pt")
                ddim_inv_latent = ddim_inversion_long(
                    validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                    num_inv_steps=validation_data.num_inv_steps, prompt="",window_size=clip_length,stride=validation_data.stride, pixel_values=pixel_values, mask=masked_pixel_values, control=control)[-1].to(weight_dtype)
                torch.save(ddim_inv_latent, inv_latents_path)
            samples = []
            for idx, prompt in enumerate(validation_data.prompts):
                with torch.autocast("cuda"):
                    validation_multidata = copy.deepcopy(validation_data)
                    validation_multidata.video_length = video_length
                    sample = validation_pipeline.gen_long(prompt,pixel_values, masked_pixel_values, generator=generator, latents=ddim_inv_latent,window_size=validation_data.video_length,control=control,
                                             **validation_multidata).videos
                save_videos_grid(sample, f"{output_dir}/samples/sample/{prompt}.gif")
                samples.append(sample)
            samples = torch.concat(samples)
            save_path = f"{output_dir}/samples/sample.gif"
            save_videos_grid(samples, save_path)
            logger.info(f"Saved samples to {save_path}")
            break
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuning-free-inpaint/girl-glass.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"max memory allocated: {max_memory_allocated:.3f} GB.")