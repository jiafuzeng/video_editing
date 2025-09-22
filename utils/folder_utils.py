import os
import uuid
import tempfile
import shutil
import re
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


def sanitize_filename(filename: str) -> str:
    """清理文件名，保留中文、字母、数字和下划线，过滤特殊字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 清理后的文件名
    """
    # 移除文件扩展名
    name, ext = os.path.splitext(filename)
    
    # 保留中文、字母、数字和下划线，其他字符替换为下划线
    # 中文字符范围：\u4e00-\u9fff
    sanitized_name = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9_]', '_', name)
    
    # 确保不以数字开头
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = f"file_{sanitized_name}"
    
    # 确保不为空
    if not sanitized_name:
        sanitized_name = "file"
    
    return f"{sanitized_name}{ext}"


def create_sanitized_temp_folder(input_folder_path: str) -> tuple[str, str]:
    """创建包含清理后文件名的临时文件夹
    
    Args:
        input_folder_path: 输入文件夹路径
        
    Returns:
        tuple[str, str]: (临时文件夹路径, 文件名映射字典的JSON字符串)
    """
    import json
    
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp(prefix="sanitized_videos_")
    
    # 文件名映射：原文件名 -> 新文件名
    filename_mapping = {}
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, filename)
        
        # 只处理文件（不是目录）
        if os.path.isfile(file_path):
            # 检查是否是视频文件
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                # 生成清理后的文件名
                sanitized_filename = sanitize_filename(filename)
                
                # 确保新文件名唯一
                counter = 1
                original_sanitized = sanitized_filename
                while sanitized_filename in filename_mapping.values():
                    name, ext = os.path.splitext(original_sanitized)
                    sanitized_filename = f"{name}_{counter}{ext}"
                    counter += 1
                
                # 创建软连接到临时文件夹
                temp_file_path = os.path.join(temp_dir, sanitized_filename)
                try:
                    os.symlink(file_path, temp_file_path)
                except (OSError, NotImplementedError):
                    # 如果软连接失败（如Windows系统或权限不足），回退到复制文件
                    shutil.copy2(file_path, temp_file_path)
                
                # 记录映射关系
                filename_mapping[filename] = sanitized_filename
    
    # 将映射关系保存为JSON字符串
    mapping_json = json.dumps(filename_mapping, ensure_ascii=False, indent=2)
    
    return temp_dir, mapping_json


def cleanup_temp_folder(temp_dir: str):
    """清理临时文件夹
    
    Args:
        temp_dir: 临时文件夹路径
    """
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass  # 忽略清理错误
