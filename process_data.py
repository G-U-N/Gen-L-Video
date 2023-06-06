import sys
import argparse
from functools import partial
import os
import shlex
import subprocess
import pathlib


base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'

names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]

for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='annotator/ckpts/')
from annotator.canny import apply_canny
from annotator.hed import apply_hed, nms
from annotator.midas import apply_midas
from annotator.mlsd import apply_mlsd
from annotator.openpose import apply_openpose
from annotator.uniformer import apply_uniformer
from annotator.util import HWC3, resize_image
from PIL import Image

from moviepy.editor import VideoFileClip,ImageSequenceClip
import cv2
import torch
import einops
import numpy as np
# Download necessary backbone weights.

def check_imgorstr(fn):
    def new_fn(*args,**kwargs):
        if isinstance(args[1],str):
            img = Image.open(args[1])
            args = *args[:1],np.array(img),*args[2:]
        return fn(*args,**kwargs)
    return new_fn 

class ProcessTools:
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_canny(self, input_image, image_resolution, low_threshold, high_threshold, **kwargs):

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        return detected_map
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_hough(self, input_image, image_resolution, detect_resolution, value_threshold,
                      distance_threshold, **kwargs):

        input_image = HWC3(input_image)
        detected_map = apply_mlsd(resize_image(input_image, detect_resolution),
                                  value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        return detected_map
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_hed(self, input_image, image_resolution, detect_resolution, **kwargs):

        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)

        return detected_map
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_scribble(self, input_image, image_resolution, **kwargs):

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        return detected_map
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_scribble_interactive(self, input_image, image_resolution, **kwargs):

        img = resize_image(HWC3(input_image['mask'][:, :, 0]),
                           image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) > 127] = 255

        return detected_map

    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_fake_scribble(self, input_image, image_resolution, detect_resolution, **kwargs):

        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        return detected_map

    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_pose(self, input_image, image_resolution, detect_resolution, **kwargs):

        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(
            resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        
        return detected_map
    
    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_seg(self, input_image, image_resolution, detect_resolution, **kwargs):

        input_image = HWC3(input_image)
        detected_map = apply_uniformer(
            resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        return detected_map

    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_depth(self, input_image, image_resolution, detect_resolution, **kwargs):

        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(
            resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        
        return detected_map

    @classmethod
    @torch.inference_mode()
    @check_imgorstr
    def process_normal(self, input_image, image_resolution, detect_resolution, bg_threshold, **kwargs):

        input_image = HWC3(input_image)
        _, detected_map = apply_midas(resize_image(input_image,
                                                   detect_resolution),
                                      bg_th=bg_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        
        return detected_map

class VideoProcessor:
    def __init__(self, v_path, t_path, c_path, control_task,trim_value=3, fps = 30, split=0):
        self.v_path = v_path
        self.c_path = c_path
        self.t_path = t_path
        self.fps = fps
        self.control_task = control_task
        self.trim_value = trim_value
        self.controls = ["canny","hough", "hed","scribble","fake_scribble","pose","seg","depth","normal"]
        self.tmp_path = f"tmp-{trim_value}-{split}"

        if not os.path.exists(self.t_path):
            os.makedirs(self.t_path)
        for control in self.controls:
            if not os.path.exists(os.path.join(self.c_path,control)):
                os.makedirs(os.path.join(self.c_path,control))
        if not os.path.exists(f"../{self.tmp_path}"):
            os.makedirs(f"../{self.tmp_path}")

    def process(self):
        for video_name in os.listdir(self.v_path):
            if video_name in os.listdir(self.t_path):
                print(f"already process {video_name},skip")
                continue
            try:
                self._process(video_name)
            except:
                print(f"process {video_name} error, continue process next!")
                continue


    def get_frames(self, video_in):
        frames = []
        clip = VideoFileClip(os.path.join(self.v_path,video_in))
        

        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile(f"../{self.tmp_path}/video_resized.mp4", fps=self.fps)

        print("video resized to 512 height")

        cap= cv2.VideoCapture(f"../{self.tmp_path}/video_resized.mp4")

        fps = cap.get(cv2.CAP_PROP_FPS)
        print("video fps: " + str(fps))
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False or i > self.trim_value * self.fps +10:
                break
            frame = resize_image(frame,512)
            cv2.imwrite(f'../{self.tmp_path}/'+str(i)+'.jpg',frame)
            frames.append(f'../{self.tmp_path}/'+str(i)+'.jpg')
            i+=1

        cap.release()
        print("broke the video into frames")

        return frames, fps

    def _create_video(self, frames, video_namae):
        print("building video result")
        clip = ImageSequenceClip(frames, fps=self.fps)
        clip.write_videofile(video_namae, fps=self.fps,codec="libx264")
        

    def _process(self, video_name):

        break_vid = self.get_frames(video_name)
        frames_list= break_vid[0]
        print("len(frames_list)",len(frames_list))
        fps = break_vid[1]
        n_frame = int(self.trim_value*fps)
        
        if n_frame >= len(frames_list):
            print("video is shorter than the cut value")
            n_frame = len(frames_list)
        
        print("set stop frames to: " + str(n_frame))
        
        kwargs = {
        "num_samples" : 1,
        "image_resolution" : 512,
        "detect_resolution" : 512,
        "low_threshold" : 100,
        "high_threshold" : 200,
        "value_threshold" : 0.1,
        "distance_threshold" : 0.1,
        "bg_threshold" : 0.4,
        }
        t_frames = frames_list[0:int(n_frame)]
        for control in self.controls:
            c_frames = []
            for t_frame in t_frames:
                c_frame = getattr(ProcessTools,"process_"+control)(t_frame,**kwargs)
                print("shape:",c_frame.shape)
                c_frames.append(c_frame)
            print(c_frames[0].shape)
            self._create_video(c_frames, os.path.join(self.c_path,control,video_name))
        self._create_video(t_frames, os.path.join(self.t_path,video_name))


parser = argparse.ArgumentParser()
parser.add_argument("--v_path",default="../data")
parser.add_argument("--t_path",default="../t_data")
parser.add_argument("--c_path",default="../c_data")
parser.add_argument("--control_task",default="multi")
parser.add_argument("--trim_value",default=100,type=int)
parser.add_argument("--fps",default=10,type=int)
parser.add_argument("--split",default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    '''
    split for supporting multiprocessor.
    '''
    videoprocessor = VideoProcessor(args.v_path,args.t_path,args.c_path,args.control_task,args.trim_value,args.fps, args.split)
    videoprocessor.process()
