import torch
import os
import sys
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management
from comfy.cli_args import args
import folder_paths
import folder_paths
from .AC_FUN import AC_FUN

EXAMPLE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'example.png')
# print(example)

# get controlnet_preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)  

# script_dir_1 = os.path.dirname(os.path.abspath(__file__))
absolute_path_1 = ('comfyui_controlnet_aux')
absolute_path_2 = ('node_wrappers')
file_path_1 = os.path.join(parent_dir,absolute_path_1,absolute_path_2)

folder_path = file_path_1

files = os.listdir(folder_path)
# print(files)

py_files = [file for file in files if file.endswith(".py")]
# print(py_files)

new_str = ' '.join(py_files)
# print(new_str) 

my_list = new_str.split('.py')
length = len(my_list)

Controlnet_Process = my_list[0:24]
# print(Controlnet_Process)

Switch = ['ADD_CN','NONE']



class AC_Super_Controlnet_Design(AC_FUN):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "Switch":(Switch,),
                             "control_net_process": (Controlnet_Process,),
                             "control_net": (folder_paths.get_filename_list("controlnet"), ),
                             "image": (sorted(files), {"image_upload": True}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    def load_controlnet(self, control_net):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return controlnet
    
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return image
    
    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent):
        control_net = self.load_controlnet(control_net)
        image = self.load_image(image) or EXAMPLE
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])