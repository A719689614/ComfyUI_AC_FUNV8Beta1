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
# import latent_preview

from .AC_FUN import AC_FUN


class AC_Super_Controlnet(AC_FUN):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
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
        image = self.load_image(image)
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


# 超级模型节点
class AC_Super_Checkpoint(AC_FUN):
    AC_list = ['model1', 'model2', 'model3', 'model4']
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { 
            "ckpt_name_1": (folder_paths.get_filename_list("checkpoints"), ),
            "ckpt_name_2": (folder_paths.get_filename_list("checkpoints"), ),
            "ckpt_name_3": (folder_paths.get_filename_list("checkpoints"), ),
            "ckpt_name_4": (folder_paths.get_filename_list("checkpoints"), ),
            "Select": (self.AC_list,),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "super_checkpoint"



    def super_checkpoint(self, ckpt_name_1,ckpt_name_2,ckpt_name_3,ckpt_name_4,Select, output_vae=True, output_clip=True,):
        if Select == 'model1':
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name_1)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return out[:3]
        if Select == 'model2':
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name_2)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return out[:3]
        if Select == 'model3':
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name_3)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return out[:3]
        if Select == 'model4':
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name_4)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return out[:3]



class AC_Super_Loras(AC_FUN):
    def __init__(self):
        self.loaded_lora = None
    lora_list = ['lora_1', 'lora_2', 'lora_3', 'lora_4']
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name_1": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_1": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "lora_name_2": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_2": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "lora_name_3": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_3": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip_3": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "lora_name_4": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_4": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip_4": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "Select":(self.lora_list,)
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    def load_lora(self, model, clip,
                  lora_name_1, strength_model_1, strength_clip_1,
                  lora_name_2, strength_model_2, strength_clip_2,
                  lora_name_3, strength_model_3, strength_clip_3,
                  lora_name_4, strength_model_4, strength_clip_4,
                  Select,
                  
                  ):
            if Select == "lora_1":
                 lora_name, strength_model, strength_clip = lora_name_1, strength_model_1, strength_clip_1
            if Select == "lora_2":
                 lora_name, strength_model, strength_clip = lora_name_2, strength_model_2, strength_clip_2
            if Select == "lora_3":
                 lora_name, strength_model, strength_clip = lora_name_3, strength_model_3, strength_clip_3
            if Select == "lora_4":
                 lora_name, strength_model, strength_clip = lora_name_4, strength_model_4, strength_clip_4

            
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    self.loaded_lora = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)

# lora merge
class AC_Lora_super(AC_FUN):
    loaded_lora = None
    boolean = ['Merge', 'Divide']
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name_1": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "lora_name_2": (folder_paths.get_filename_list("loras"), ),
                              "strength_model_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "boolean":(self.boolean,)
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "AC_Merge"

    @classmethod
    def load_lora(cls, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if cls.loaded_lora is not None:
            if cls.loaded_lora[0] == lora_path:
                lora = cls.loaded_lora[1]
            else:
                temp = cls.loaded_lora
                cls.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            cls.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)
    
    def AC_Merge(self, model, clip,
                 lora_name_1, strength_model_1, strength_clip_1,
                 lora_name_2, strength_model_2, strength_clip_2,
                 boolean
                 ):
        if boolean == 'Divide':
            result = self.__class__.load_lora(model, clip,lora_name_1, strength_model_1, strength_clip_1)
            return result
        if boolean == 'Merge':
            result_1 = self.__class__.load_lora(model, clip,lora_name_1, strength_model_1, strength_clip_1)
            model_s = result_1[0]
            clip_s = result_1[1]
            result_2 = self.__class__.load_lora(model_s, clip_s,lora_name_2, strength_model_2, strength_clip_2)
            return result_2
