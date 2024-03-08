from .super_ac import *
from .super_ac_ksample import *
from .AC_combos import Su_Checkpoint,Su_Lora
from .AC_Large import AC_FUN_SUPER_LARGE
NODE_CLASS_MAPPINGS = {
     "⬛(TODO)AC_Super_Come_Ckpt":Su_Checkpoint,
     "⬜(TODO)AC_Super_Come_Lora":Su_Lora,
     "🎰AC_Super_Controlnet": AC_Super_Controlnet,
     "🈵AC_Super_Checkpoint": AC_Super_Checkpoint,
     "🈵AC_Super_Loras": AC_Super_Loras,
     "💼AC_Super_Lora&LCM":AC_Lora_super,
     "🚀AC_Super_KSampler":AC_Super_KSampler,
     "📈AC_Super_UpKSampler":AC_Super_UPKSampler,
     "💾AC_Super_SaveImage":AC_Super_SaveImage,
     "🎫AC_Super_PreviewImage":AC_Super_PreviewImage,
     "📟AC_Super_CKPT&LCM":Checkpoint,
     "📄AC_Super_CLIPEN":AC_CLIP_EN,
     "💶AC_Super_EmptLatent":AC_SUPER_EmptyLatent,
     "⭕AC_FUN_SUPER_LARGE":AC_FUN_SUPER_LARGE,
}