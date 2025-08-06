import torch

import os
import sys
import json
import random
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
import comfy.clip_vision
import comfy.model_management
from comfy.cli_args import args
import base64
import folder_paths
import latent_preview
from io import BytesIO
from .AC_FUN import AC_FUN
from .AC_Main import Checkpoint,AC_CLIPText

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def base64_save(base64_data):
    data = base64_data.split(",")[1]
    decoded_data = base64.b64decode(data)
    image = Image.open(BytesIO(decoded_data))
    image,mask=load_image_ts(image)
    return (image,mask)

def load_image_ts(i,white_bg=False):
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
        if white_bg==True:
            nw = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
            image[nw == 1] = 1.0
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image,mask)


MAX_RESOLUTION = 5277
EXAMPLE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'example.png')
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out




class AC_Super_KSampler(AC_FUN):
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    AC_list = ['512*512', '576*576', '640*640', '640*480', '480*640','512*768', '768*512','768*768', 
        '896*896','512*960', '512*1024', '1024*1024','1024*512', '768*1280', '1280*768','720*1280', 
        '1280*720', '1024*1536','1536*1024', '1080*1920', '1920*1080', '2048*1536' ,'1536*2048', 
        '3200*2048', '2048*3200','3840*1600', '1600*3840', '3840*2400', '2400*3840', '3480*2600',
        '2600*3480', '4096*3072', '3072*4096', '4515*3386', '5120*3840', '5120*4090','6400*4800'
        ]
    Dispatch = ["Custom","All_ready"]
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                    
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,{"default":"lcm"} ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "clip": ("CLIP", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "positive": ("STRING", {"multiline": True,
                                            "default":"1girl, skirt, hair ornament, long hair, glasses, solo, twin braids, braid, sitting, purple eyes, looking at viewer, page number, shirt, brown hair, white shirt, collarbone, ahoge, simple background, pleated skirt, navel, open clothes, cardigan, school uniform, flower, plaid, bangs, breasts, very long hair, long sleeves, pink cardigan, plaid skirt, hair flower, clock, blush, open cardigan, hairclip, closed mouth, collared shirt, off shoulder, buttons, smile, groin, white background, medium breasts, miniskirt, twintails, ribbon, unbuttoned, artist name, shiny hair, shiny, red ribbon, bow,blond hair, long blond hair, curly hair, dressed in armor,"}), 
                    "negative": ("STRING", {"multiline": True,
                                            "default":"lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"}),
                    "resolution":(self.AC_list,),
                    "dispatch":(self.Dispatch,),
                    "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                    "height": ("INT", {"default": 768, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                     }
                }

    RETURN_TYPES = ("LATENT","CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("LATENT","Positive","Negative")
    FUNCTION = "sample"
    
    @classmethod
    def generate(cls, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return {"samples":latent}


    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, clip=None,positive=None, negative=None,width = 512, height = 512, batch_size =1,
                latent_image=None, denoise=1.0, resolution=None,dispatch=None):
        # dispatch
        if dispatch =='Custom':
            latent_image = self.generate(width, height, batch_size)
        if dispatch == 'All_ready':
            str = resolution
            list = str.split('*')
            width, height = list[0],list[1]
            width, height = int(width),int(height)
            latent_image = self.generate(width, height, batch_size)

        tokens_p = clip.tokenize(positive)
        cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
        positive = [[cond_p, {"pooled_output": pooled_p}]]

        tokens_n = clip.tokenize(negative)
        cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
        negative = [[cond_n, {"pooled_output": pooled_n}]]

        return (common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise), positive, negative)

# AC超级放大
class AC_Super_UPKSampler(AC_FUN):
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,{"default":"lcm"} ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "samples": ("LATENT",), 
                    "upscale_method": (s.upscale_methods,),
                    "crop": (s.crop_methods,),
                    "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                    "height": ("INT", {"default": 1536, "min": 16, "max": MAX_RESOLUTION, "step": 1})
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive=None, negative=None,samples=None,crop=None,
               width = None, height = None,upscale_method=None,vae=None,
                latent_image=None, denoise=0.35):
        
        s = samples.copy()
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width // 8, height // 8, upscale_method, crop)
        latent_image = s

        return (common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise),)
    
# 超级保存节点
class AC_Super_SaveImage(AC_FUN):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": 
                    {
                    #  "samples": ("LATENT", ), "vae": ("VAE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})
                     },
            "optional": 
                    {
                      "samples": ("LATENT", ), 
                     "vae": ("VAE", ),
                     "images": ("IMAGE", ),
                     },
                }

    RETURN_TYPES = ( )
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    @classmethod
    def convert_base64_to_image(cls,base64_string):
        # 解码base64字符串为字节流
        image_bytes = base64.b64decode(base64_string)
        # 创建BytesIO对象，并将字节流写入其中
        image_buffer = BytesIO(image_bytes)
        # 打开图像并返回PIL图像对象
        image = Image.open(image_buffer)
        return image

    def save_images(self, images=None, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None,samples=None,vae=None):
        if samples is not None and vae is not None:
            images = vae.decode(samples["samples"])
        else:
            images = images
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
    
        results = { "ui": { "images": results } }

        return results
        

# 超级显示节点
class AC_Super_PreviewImage(AC_Super_SaveImage,AC_FUN):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
    
    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(self):
        return {"optional":
                    {
                        "samples": ("LATENT", ), 
                        "vae": ("VAE", ),
                        "images": ("IMAGE", ),
                        },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                } 

# 超级生成器
class actesting(Checkpoint): 
    pass

# 超级文本解码器
class AC_CLIP_EN(AC_CLIPText):
    pass

# 超级潜空间变量
class AC_SUPER_EmptyLatent(AC_FUN):
    def __init__(self, device="cpu"):
        self.device = device
    AC_list = ['512*512', '576*576', '640*640', '640*480', '480*640','512*768', '768*512','768*768', 
        '896*896','512*960', '512*1024', '1024*1024','1024*512', '768*1280', '1280*768','720*1280', 
        '1280*720', '1024*1536','1536*1024', '1080*1920', '1920*1080', '2048*1536' ,'1536*2048', 
        '3200*2048', '2048*3200','3840*1600', '1600*3840', '3840*2400', '2400*3840', '3480*2600',
        '2600*3480', '4096*3072', '3072*4096', '4515*3386', '5120*3840', '5120*4090','6400*4800'
        ]
    boolean = ["Custom","All_ready"]
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": { 
            "Resolution":(self.AC_list,),
            "Boolean":(self.boolean,),
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }}
    RETURN_TYPES = ("LATENT","INT","INT")
    RETURN_NAMES = ("LATENT","WIDTH","HEIGHT")
    FUNCTION = "generate"

    def generate(self, Resolution, Boolean, width, height, batch_size=1):
        if Boolean == "Custom":
            latent = torch.zeros([batch_size, 4, height // 8, width // 8])
            return ({"samples":latent},width, height, )
        if Boolean == "All_ready":
            str = Resolution
            list = str.split('*')
            width, height = list[0],list[1]
            width, height = int(width),int(height)
            # new_str =f"{width},{height}"
            latent = torch.zeros([batch_size, 4, height // 8, width // 8])
            return ({"samples":latent}, width, height)