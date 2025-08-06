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
import latent_preview
from .AC_FUN import AC_FUN
from .image_tools import pil2tensor, tensor2pil

MAX_RESOLUTION = 5277
EXAMPLE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'example.png')

class VAEEncode:
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def encode_im(self, vae, pixels):
        pixels = self.vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:,:,:,:3])
        return {"samples":t}

# AC_FUN_SUPER_LARGE
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out

class AC_FUN_SUPER_LARGE(AC_FUN,VAEEncode):
    def __init__(self, device="cpu"):
        self.device = device
    loaded_lora = None
    boolean = ['Single', 'Double', 'Triple']
    AC_list = ['512*512', '576*576', '640*640', '640*480', '480*640','512*768', '768*512','768*768', 
        '896*896','512*960', '512*1024', '1024*1024','1024*512', '768*1280', '1280*768','720*1280', 
        '1280*720', '1024*1536','1536*1024', '1080*1920', '1920*1080', '2048*1536' ,'1536*2048', 
        '3200*2048', '2048*3200','3840*1600', '1600*3840', '3840*2400', '2400*3840', '3480*2600',
        '2600*3480', '4096*3072', '3072*4096', '4515*3386', '5120*3840', '5120*4090','6400*4800'
        ]
    Dispatch = ["Custom","All_ready"]
    Select_Model = ['Prompt_to_Image', 'Image_to_Image']
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if len(files) == 0:
            files.append(EXAMPLE)
        return {"required": { 
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "lora_name_1": (folder_paths.get_filename_list("loras"), ),
            "strength_model_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_2": (folder_paths.get_filename_list("loras"), ),
            "strength_model_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_3": (folder_paths.get_filename_list("loras"), ),
            "strength_model_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "boolean":(self.boolean,),
            "positive": ("STRING", {"multiline": True, "default":"(best quality), ((masterpiece)), (highres), illustration, original, extremely detailed,1girl, solo, kashima \(kancolle\), gloves, breasts, hat, epaulettes, grey hair, large breasts, white gloves, smile, military uniform, military, uniform, twintails, tsurime, white background, beret, simple background, looking at viewer, wavy hair, buttons, blush, long hair, frilled sleeves, purple eyes, upper body, frills, long sleeves, neckerchief, red neckerchief"}),
            "negative": ("STRING", {"multiline": True, "default":"(worst quality, low quality, blurry, bad eye, ),(wrong hand, bad anatomy, wrong anatomy, ),(cgi, illustration, cartoon, poorly drawn, watermark),head out of frame,"}),
            "resolution":(self.AC_list,),
            "dispatch":(self.Dispatch,),
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,{"default":"lcm"} ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "select_model":(self.Select_Model,),
            "image": (files, {"image_upload": True})
                             }}
    RETURN_TYPES = ("MODEL","LATENT","CONDITIONING","CONDITIONING","INT","INT","VAE")
    RETURN_NAMES = ("MODEL","LATENT","Positive","Negative","width","height","VAE")

    FUNCTION = "super_large"

    @classmethod
    def load_checkpoint(cls, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        if len(out) >= 3:
            model, clip, vae = out[:3]  
        else:
            model = out[0] if len(out) >= 1 else None
            clip = out[1] if len(out) >= 2 else None
            vae = out[2] if len(out) >= 3 else None
        return model, clip, vae
    
    @classmethod
    def load_lora(cls, ckpt_name=None,lora_name=None, strength_model=1, strength_clip=1):
        model, clip, vae = cls.load_checkpoint(ckpt_name)

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
        return (model_lora, clip_lora, vae)
    
    @classmethod
    def add_lora(cls, model, clip, lora_name, strength_model, strength_clip):
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

    @classmethod
    def encode(cls, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    @classmethod
    def generate(cls, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return {"samples":latent}

    @classmethod
    def load_image(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path) or Image.open(EXAMPLE)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image
    
    @classmethod
    def ac_tensor2pil(cls,image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def super_large(self,
            seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image=None, denoise=1.0,
            ckpt_name=None, 
            lora_name_1=None, strength_model_1=1, strength_clip_1=1,
            lora_name_2=None, strength_model_2=1, strength_clip_2=1,
            lora_name_3=None, strength_model_3=1, strength_clip_3=1,
            width=512, height=768, batch_size=1,clip=None,resolution=None,dispatch=None,boolean=None,select_model=None,
            image=None,
                    ):
        if boolean == "Single":
            model_lora, clip_lora, vae = self.load_lora(ckpt_name, lora_name_1, strength_model_1, strength_clip_1)
            model = model_lora
            clip = clip_lora
            vae = vae
        if boolean == "Double":
            model_lora_1, clip_lora_1, vae = self.load_lora(ckpt_name, lora_name_1, strength_model_1, strength_clip_1)
            model_lora, clip_lora, = self.add_lora(model_lora_1, clip_lora_1, lora_name_2, strength_model_2, strength_clip_2)
            model = model_lora
            clip = clip_lora
            vae = vae
        if boolean == "Triple":
            model_lora_1, clip_lora_1, vae = self.load_lora(ckpt_name, lora_name_1, strength_model_1, strength_clip_1)
            model_lora_2, clip_lora_2, = self.add_lora(model_lora_1, clip_lora_1, lora_name_2, strength_model_2, strength_clip_2)
            model_lora_3, clip_lora_3, = self.add_lora(model_lora_2, clip_lora_2, lora_name_3, strength_model_3, strength_clip_3)
            model = model_lora_3
            clip = clip_lora_3
            vae = vae


        positive = self.encode(clip,positive)
        negative = self.encode(clip,negative)
        
        # dispatch
        if select_model == 'Prompt_to_Image':
            if dispatch =='Custom':
                latent_image = self.generate(width, height, batch_size)
            if dispatch == 'All_ready':
                str = resolution
                list = str.split('*')
                width, height = list[0],list[1]
                width, height = int(width),int(height)
                latent_image = self.generate(width, height, batch_size)

        if select_model == 'Image_to_Image':
            image = self.load_image(image)
            # if not image:
            #     image = self.load_image(EXAMPLE)
            model, clip, vae = self.load_checkpoint(ckpt_name)
            
            vae_encode = self.vae_encode_crop_pixels(image)
            latent_image = self.encode_im(vae,vae_encode)

        
        return (model, common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise), positive, negative, width, height, vae)

# AC_FUN_SUPER_LARGE_AERA