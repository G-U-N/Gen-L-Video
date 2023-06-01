import random
import decord
import torch
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import numpy as np
import PIL
import torchvision

class GLVDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            stride: int = 4,
            control_path: str = None,
            control_channels = 3,
    ):
        self.video_path = video_path
        if isinstance(prompt,str):
            self.prompt = prompt
        else:
            self.prompt = []
            gap = n_sample_frames//stride -1
            for i in range(len(prompt)-1):
                self.prompt.append(prompt[i])
                for j in range(gap):
                    if j<=gap//2:
                        self.prompt.append(prompt[i])
                    else:
                        self.prompt.append(prompt[i+1])
            self.prompt.append(prompt[-1])
        self.prompt_ids = None
        self.null_prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.stride = stride
        self.vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        self.sample_index = list(range(self.sample_start_idx, len(self.vr), self.sample_frame_rate))        
        self.full_video = rearrange(self.vr.get_batch(self.sample_index),"f h w c -> f c h w")
        self.len_video = len(self.sample_index) 
        self.len_video = self.len_video-self.len_video%self.n_sample_frames
        self.sample_index = self.sample_index[:self.len_video]
        self.len_dataset = (self.len_video - self.n_sample_frames)//self.stride + 1
        # self.trsf = ColorJitter(0.3,0.3)
        self.control_path = control_path
        self.control_channels = control_channels
        if control_path is not None:
            self.control_vr = decord.VideoReader(control_path, width=self.width, height=self.height)
            self.full_control_video = rearrange(self.control_vr.get_batch(self.sample_index),"f h w c -> f c h w")
            assert control_channels ==1 or control_channels == 3
            if control_channels == 1:
                self.full_control_video = self.full_control_video[:,0:1]  
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        prompt_ids = self.prompt_ids if isinstance(self.prompt,str) else self.prompt_ids[index]
        index = index * self.stride
        sample_index = self.sample_index[index:index+self.n_sample_frames]
        video = self.vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        if self.control_path is not None:
            control_video = self.control_vr.get_batch(sample_index)
            control_video = rearrange(control_video, "f h w c -> f c h w")
            if self.control_channels == 1:
                control_video = control_video[:,0:1]
        
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
            "video_length": self.len_video,
            "full_video": (self.full_video / 127.5 -1.0),
            "clip_id": index,
        }
        if self.null_prompt_ids is not None:
            example["null_prompt_ids"] = self.null_prompt_ids
        
        if self.control_path is not None:
            example["full_control_video"] = self.full_control_video / 255.0
            example["control_video"] = control_video / 255.

        return example




class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image) or isinstance(clip[0], torch.Tensor):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img.unsqueeze(0))

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        jittered_clip = torch.cat(jittered_clip)
        return jittered_clip


