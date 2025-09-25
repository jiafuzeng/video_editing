import os
import glob
from pathlib import Path
import logging as logger
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
            logger.info(f"[VideoCropNode] 开始裁切 | input_folder={input_folder} | output_prefix={output_folder_prefix} | 区域=({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
            # 解析输入路径（支持相对路径和绝对路径）
            input_folder_path = resolve_path(input_folder)
            logger.info(f"[VideoCropNode] 解析后的输入路径: {input_folder_path}")
            
            # 验证输入路径是否存在
            if not os.path.exists(input_folder_path):
                logger.warning(f"[VideoCropNode] 输入路径不存在，终止: {input_folder_path}")
                return ("",)
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            unique_folder_name = generate_unique_folder_name(output_folder_prefix, output_dir)
            output_path = os.path.join(output_dir, unique_folder_name)
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"[VideoCropNode] 输出目录已创建/存在: {output_path}")
            
            # 创建包含清理后文件名的临时文件夹
            temp_dir, filename_mapping = create_sanitized_temp_folder(input_folder_path)
            logger.info(f"[VideoCropNode] 临时目录创建完成: {temp_dir} | 规范化文件数: {len(filename_mapping)}")
            
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
                            logger.info(f"[VideoCropNode] 处理: {video_file} -> {output_file}")
                            
                            # 计算裁切宽度和高度
                            crop_width = crop_x2 - crop_x1
                            crop_height = crop_y2 - crop_y1
                            logger.info(f"[VideoCropNode] 裁切参数: width={crop_width}, height={crop_height}, x1={crop_x1}, y1={crop_y1}")
                            
                            # 检查原视频是否有音效
                            has_audio = False
                            try:
                                probe = ffmpeg.probe(video_file)
                                audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                                has_audio = len(audio_streams) > 0
                                logger.info(f"[VideoCropNode] 音频检测: {'有音频' if has_audio else '无音频'}")
                            except Exception:
                                has_audio = False
                                logger.warning("[VideoCropNode] 音频检测失败，按无音频处理")
                            
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
                                logger.info("[VideoCropNode] 完成裁切并保留音频")
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
                                logger.info("[VideoCropNode] 完成裁切（无音频输出）")
                            
                            processed_count += 1
                            logger.info(f"[VideoCropNode] 成功: {output_file}")
                            
                        except Exception as e:
                            logger.error(f"[VideoCropNode] 处理失败: {video_file} | 错误: {e}")
                            continue
                
                if processed_count == 0:
                    logger.warning("[VideoCropNode] 未处理任何视频文件")
                    return ("",)  # 没有可处理的视频时返回空字符串
                else:
                    logger.info(f"[VideoCropNode] 处理完成，共输出 {processed_count} 个文件 | 输出目录: {output_path}")
                    return (output_path,)
                    
            finally:
                # 清理临时文件夹
                logger.info(f"[VideoCropNode] 清理临时目录: {temp_dir}")
                cleanup_temp_folder(temp_dir)
                
        except ValueError as e:
            # 输入验证错误
            logger.error(f"[VideoCropNode] 输入参数错误: {e}")
            return ("",)

class VideoPreviewNode(ComfyNodeABC):
    """
    视频预览节点
    支持上传视频文件或指定视频路径，提供视频预览功能
    """
    
    DESCRIPTION = "视频预览节点 - 支持上传或指定视频路径进行预览"
    CATEGORY = "video/video_editing"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # 采用内置上传：列出 input 目录下的视频，并启用 video_upload
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"]) or []
        files = sorted(files)
        return {
            "required": {
                "video_path": (files, {"video_upload": True, "tooltip": "选择或上传视频 (保存到 input 目录)"}),
            },
            "optional": {
                # 这四个参数用于前端写入坐标（在前端被隐藏）
                "crop_x1": (IO.INT, {"default": 0, "min": 0, "max": 16384, "tooltip": "左上X"}),
                "crop_y1": (IO.INT, {"default": 0, "min": 0, "max": 16384, "tooltip": "左上Y"}),
                "crop_x2": (IO.INT, {"default": 0, "min": 0, "max": 16384, "tooltip": "右下X"}),
                "crop_y2": (IO.INT, {"default": 0, "min": 0, "max": 16384, "tooltip": "右下Y"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING, IO.INT, IO.INT, IO.INT, IO.INT)
    RETURN_NAMES = ("video_path", "crop_x1", "crop_y1", "crop_x2", "crop_y2")
    FUNCTION = "preview_video"
    OUTPUT_NODE = False
    
    def preview_video(self, video_path: str, crop_x1: int = 0, crop_y1: int = 0, crop_x2: int = 0, crop_y2: int = 0):
        """
        预览视频文件并获取用户选择的区域坐标
        
        Args:
            video_path: 视频文件路径（来自文件选择器）
            
        Returns:
            tuple: (视频路径, 左上角X坐标, 左上角Y坐标, 右下角X坐标, 右下角Y坐标)
        """
        try:
            # 如果路径为空，返回空值
            if not video_path or video_path.strip() == "":
                return ("", 0, 0, 0, 0)
            
            # 优先从 input 目录解析（兼容上传/下拉选择的文件名）
            resolved = None
            try:
                # 若为上传/下拉的文件名，优先按注解规则解析
                # 与 comfy_extras.nodes_video.LoadVideo 一致
                candidate = folder_paths.get_annotated_filepath(video_path)
                if os.path.exists(candidate):
                    resolved = candidate
            except Exception:
                resolved = None
            if not resolved:
                # 解析路径，兼容相对与绝对路径
                resolved = resolve_path(video_path)
            
            # 验证视频文件是否存在
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 验证是否为视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
            file_extension = os.path.splitext(resolved)[1].lower()
            if file_extension not in video_extensions:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            # 返回解析后的路径和坐标
            return (resolved, crop_x1, crop_y1, crop_x2, crop_y2)
            
        except Exception as e:
            raise Exception(f"视频预览失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoCropNode": VideoCropNode,
    "VideoPreviewNode": VideoPreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCropNode": "视频画幅裁切",
    "VideoPreviewNode": "视频预览"
}