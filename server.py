import os
from aiohttp import web
import folder_paths
import server as comfy_server


def _get_dir_by_type(dir_type: str):
    if not dir_type:
        dir_type = "input"
    if dir_type == "input":
        base = folder_paths.get_input_directory()
    elif dir_type == "temp":
        base = folder_paths.get_temp_directory()
    elif dir_type == "output":
        base = folder_paths.get_output_directory()
    else:
        base = None
    return base, dir_type

routes = comfy_server.PromptServer.instance.routes

@routes.get("/list_dir")
async def list_dir_direct(request: web.Request):
    # 复用上面的实现
    dir_type = "input"
    subfolder = request.rel_url.query.get("subfolder", "")

    base, actual_type = _get_dir_by_type(dir_type)
    if base is None:
        return web.Response(status=400)

    full_dir = os.path.abspath(os.path.join(base, os.path.normpath(subfolder)))
    try:
        if os.path.commonpath((base, full_dir)) != base:
            return web.Response(status=403)
    except Exception:
        return web.Response(status=403)

    if not os.path.exists(full_dir) or not os.path.isdir(full_dir):
        return web.json_response({"type": actual_type, "subfolder": subfolder, "subfolders": [], "files": []})

    try:
        entries = os.listdir(full_dir)
    except Exception:
        return web.json_response({"type": actual_type, "subfolder": subfolder, "subfolders": [], "files": []})

    subfolders = []
    files = []
    for name in sorted(entries):
        p = os.path.join(full_dir, name)
        if os.path.isdir(p):
            subfolders.append(name)
        elif os.path.isfile(p):
            files.append(name)

    return web.json_response({
        "type": actual_type,
        "subfolder": subfolder,
        "subfolders": subfolders,
        "files": files,
    })