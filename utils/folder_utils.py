import os
import uuid
import folder_paths


def resolve_path(path: str) -> str:
    """解析路径，支持相对路径和绝对路径
    
    Args:
        path: 输入路径（相对或绝对）
        
    Returns:
        str: 解析后的绝对路径
    """
    if not path or not path.strip():
        return ""
    
    path = path.strip()
    
    # 如果是绝对路径，直接返回
    if os.path.isabs(path):
        return path
    
    # 相对路径：相对于ComfyUI的input目录
    try:
        input_dir = folder_paths.get_input_directory()
        return os.path.join(input_dir, path)
    except:
        # 如果无法获取input目录，使用当前工作目录
        return os.path.abspath(path)


def generate_unique_folder_name(prefix: str, output_dir: str) -> str:
    """使用UUID生成唯一的文件夹名称
    
    Args:
        prefix: 文件夹前缀
        output_dir: 输出目录路径
        
    Returns:
        str: 唯一的文件夹名称
    """
    # 生成UUID并截取前8位
    unique_id = str(uuid.uuid4())[:8]
    folder_name = f"{prefix}_{unique_id}"
    
    # 验证文件夹是否存在，如果存在则重新生成
    while os.path.exists(os.path.join(output_dir, folder_name)):
        unique_id = str(uuid.uuid4())[:8]
        folder_name = f"{prefix}_{unique_id}"
    
    return folder_name
