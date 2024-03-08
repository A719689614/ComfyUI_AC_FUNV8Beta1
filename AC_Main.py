from .AC_FUN import AC_FUN
import folder_paths
import comfy.sd
import comfy.clip_vision
from comfy.cli_args import args

class Checkpoint(AC_FUN):
    def __init__(self):
        self.loaded_lora = None
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "lora_name": (folder_paths.get_filename_list("loras"), ),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "tips":("STRING", {"default":"AC_FUN超级模型&LORA"}),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_lora"

    
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
    
    
    def load_lora(self, ckpt_name=None,lora_name=None, strength_model=1, strength_clip=1,tips=None):
        model, clip, vae = self.__class__.load_checkpoint(ckpt_name)

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
        return (model_lora, clip_lora, vae)



class AC_CLIPText(AC_FUN):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "Positive": ("STRING", {"multiline": True, "default":"A girl, squat, cat ear headphones, pleated skirt, white suspender top, best masterpiece, best quality, high resolution"}),
            "Negative": ("STRING", {"multiline": True, "default":"(worst quality, low quality, blurry, bad eye, ),(wrong hand, bad anatomy, wrong anatomy, ),(cgi, illustration, cartoon, poorly drawn, watermark),head out of frame,"}),
            "tips":("STRING",{"multiline":True,"default":"AC_FUN_STEXT"}) 
                             }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("Positive","Negative")

    FUNCTION = "ac_encode"


    @classmethod
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    def ac_encode(self, clip, Positive, Negative,tips=None):
        Positive = self.__class__.encode(clip,Positive)
        Negative = self.__class__.encode(clip,Negative)
        return (Positive, Negative)


