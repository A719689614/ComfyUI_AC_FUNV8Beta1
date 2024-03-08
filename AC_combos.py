import glob
import os
from nodes import LoraLoader, CheckpointLoaderSimple
import folder_paths
from server import PromptServer
from folder_paths import get_directory_by_type
from aiohttp import web
import shutil
from .AC_FUN import AC_FUN


@PromptServer.instance.routes.get("/pysssss/view/{name}")
async def view(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    image_path = folder_paths.get_full_path(
        type, name)
    if not image_path:
        return web.Response(status=404)

    filename = os.path.basename(image_path)
    return web.FileResponse(image_path, headers={"Content-Disposition": f"filename=\"{filename}\""})


@PromptServer.instance.routes.post("/pysssss/save/{name}")
async def save_preview(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    body = await request.json()

    dir = get_directory_by_type(body.get("type", "output"))
    subfolder = body.get("subfolder", "")
    full_output_folder = os.path.join(dir, os.path.normpath(subfolder))

    if os.path.commonpath((dir, os.path.abspath(full_output_folder))) != dir:
        return web.Response(status=400)

    filepath = os.path.join(full_output_folder, body.get("filename", ""))
    image_path = folder_paths.get_full_path(type, name)
    image_path = os.path.splitext(
        image_path)[0] + os.path.splitext(filepath)[1]

    shutil.copyfile(filepath, image_path)

    return web.json_response({
        "image":  type + "/" + os.path.basename(image_path)
    })


@PromptServer.instance.routes.get("/pysssss/examples/{name}")
async def get_examples(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    file_path = folder_paths.get_full_path(
        type, name)
    if not file_path:
        return web.Response(status=404)
    
    file_path_no_ext = os.path.splitext(file_path)[0]
    examples = []
    if os.path.isdir(file_path_no_ext):
        examples += map(lambda t: os.path.relpath(t, file_path_no_ext),
                        glob.glob(file_path_no_ext + "/*.txt"))
   
    return web.json_response(examples)


def populate_items(names, type):
    for idx, item_name in enumerate(names):

        file_name = os.path.splitext(item_name)[0]
        file_path = folder_paths.get_full_path(type, item_name)

        if file_path is None:
            print(f"(pysssss:better_combos) Unable to get path for {type} {item_name}")
            continue

        file_path_no_ext = os.path.splitext(file_path)[0]

        for ext in ["png", "jpg", "jpeg", "preview.png"]:
            has_image = os.path.isfile(file_path_no_ext + "." + ext)
            if has_image:
                item_image = f"{file_name}.{ext}"
                break

        names[idx] = {
            "content": item_name,
            "image": f"{type}/{item_image}" if has_image else None,
        }
    names.sort(key=lambda i: i["content"].lower())


class Su_Lora(LoraLoader,AC_FUN):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        names = types["required"]["lora_name"][0]
        populate_items(names, "loras")
        return types
    
    CATEGORY  = "🔯AC_FUNV8.0"

    def load_lora(self, **kwargs):
        kwargs["lora_name"] = kwargs["lora_name"]["content"]
        return super().load_lora(**kwargs)


class Su_Checkpoint(CheckpointLoaderSimple,AC_FUN):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        names = types["required"]["ckpt_name"][0]
        populate_items(names, "checkpoints")
        return types
    
    CATEGORY  = "🔯AC_FUNV8.0"
    
    def load_checkpoint(self, **kwargs):
        kwargs["ckpt_name"] = kwargs["ckpt_name"]["content"]
        return super().load_checkpoint(**kwargs)



