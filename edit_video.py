import os
import glob
from pathlib import Path
import ffmpeg
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from .utils import generate_unique_folder_name, resolve_path, create_sanitized_temp_folder, cleanup_temp_folder

class VideoCropNode(ComfyNodeABC):
    """
    视频裁切节点
    输入文件夹路径，遍历所有视频文件，按指定坐标裁切并保存到目标文件夹
    """
    
    DESCRIPTION = "视频画幅裁切节点 - 支持相对路径和绝对路径"
    CATEGORY = "video/video_editing"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "input_folder": (IO.STRING, {"default": "", "tooltip": "输入文件夹路径（支持相对路径和绝对路径）"}),
                "output_folder_prefix": (IO.STRING, {"default": "video_crop", "tooltip": "输出文件夹前缀"}),
                "crop_x1": (IO.INT, {"default": 0, "min": 0, "max": 4096, "tooltip": "裁切区域左上角X坐标"}),
                "crop_y1": (IO.INT, {"default": 0, "min": 0, "max": 4096, "tooltip": "裁切区域左上角Y坐标"}),
                "crop_x2": (IO.INT, {"default": 1920, "min": 0, "max": 4096, "tooltip": "裁切区域右下角X坐标"}),
                "crop_y2": (IO.INT, {"default": 1080, "min": 0, "max": 4096, "tooltip": "裁切区域右下角Y坐标"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "crop_videos"
    OUTPUT_NODE = True
    
    def crop_videos(self, input_folder: str, output_folder_prefix: str, crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int):
        """
        裁切视频文件
        
        Args:
            input_folder: 输入文件夹路径（支持相对路径和绝对路径）
            output_folder_prefix: 输出文件夹前缀
            crop_x1, crop_y1: 左上角坐标
            crop_x2, crop_y2: 右下角坐标
        """
        try:
            # 解析输入路径（支持相对路径和绝对路径）
            input_folder_path = resolve_path(input_folder)
            
            # 验证输入路径是否存在
            if not os.path.exists(input_folder_path):
                return ("",)
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            unique_folder_name = generate_unique_folder_name(output_folder_prefix, output_dir)
            output_path = os.path.join(output_dir, unique_folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 创建包含清理后文件名的临时文件夹
            temp_dir, filename_mapping = create_sanitized_temp_folder(input_folder_path)
            
            try:
                # 支持的视频格式
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
                
                # 遍历所有视频文件
                processed_count = 0
                for ext in video_extensions:
                    pattern = os.path.join(temp_dir, ext)
                    video_files = glob.glob(pattern)
                
                    for video_file in video_files:
                        try:
                            # 获取文件名（不含扩展名）
                            filename = Path(video_file).stem
                            output_file = os.path.join(output_path, f"{filename}_cropped.mp4")
                            
                            # 计算裁切宽度和高度
                            crop_width = crop_x2 - crop_x1
                            crop_height = crop_y2 - crop_y1
                            
                            # 检查原视频是否有音效
                            has_audio = False
                            try:
                                probe = ffmpeg.probe(video_file)
                                audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                                has_audio = len(audio_streams) > 0
                            except Exception:
                                has_audio = False
                            
                            # 使用ffmpeg进行裁切
                            if has_audio:
                                # 保留音效的裁切
                                input_stream = ffmpeg.input(video_file)
                                video_stream = input_stream.video.filter('crop', crop_width, crop_height, crop_x1, crop_y1)
                                audio_stream = input_stream.audio
                                
                                (
                                    ffmpeg
                                    .output(video_stream, audio_stream, output_file, 
                                           vcodec='libx264', acodec='aac', 
                                           audio_bitrate='128k', preset='medium')
                                    .overwrite_output()
                                    .run(quiet=True)
                                )
                            else:
                                # 不保留音效的裁切
                                (
                                    ffmpeg
                                    .input(video_file)
                                    .video
                                    .filter('crop', crop_width, crop_height, crop_x1, crop_y1)
                                    .output(output_file, vcodec='libx264', an=None)
                                    .overwrite_output()
                                    .run(quiet=True)
                                )
                            
                            processed_count += 1
                            
                        except Exception as e:
                            continue
                
                if processed_count == 0:
                    return ("",)  # 没有可处理的视频时返回空字符串
                else:
                    return (output_path,)
                    
            finally:
                # 清理临时文件夹
                cleanup_temp_folder(temp_dir)
                
        except ValueError as e:
            # 输入验证错误
            return ("",)
# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoCropNode": VideoCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCropNode": "视频画幅裁切"
}