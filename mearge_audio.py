import os
import random
import ffmpeg
import glob
import tempfile
import datetime
import hashlib
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import resolve_path, generate_unique_folder_name
import folder_paths

# 设置logger
logger = logging.getLogger(__name__)

class AudioMerger:
    """音频合并器，支持视频与随机音频的合并"""
    
    def __init__(self):
        self.supported_audio_formats = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def get_audio_files(self, audio_folder: str) -> List[str]:
        """获取音频文件夹中的所有音频文件"""
        audio_files = []
        for ext in self.supported_audio_formats:
            pattern = os.path.join(audio_folder, f"*{ext}")
            audio_files.extend(glob.glob(pattern))
            pattern = os.path.join(audio_folder, f"*{ext.upper()}")
            audio_files.extend(glob.glob(pattern))
        return audio_files
    
    def get_video_files(self, video_folder: str) -> List[str]:
        """获取视频文件夹中的所有视频文件"""
        video_files = []
        for ext in self.supported_video_formats:
            pattern = os.path.join(video_folder, f"*{ext}")
            video_files.extend(glob.glob(pattern))
            pattern = os.path.join(video_folder, f"*{ext.upper()}")
            video_files.extend(glob.glob(pattern))
        return video_files
    
    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.error(f"获取视频时长失败: {e}")
            return 0.0
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.error(f"获取音频时长失败: {e}")
            return 0.0
    
    def create_audio_sequence(self, audio_files: List[str], target_duration: float) -> str:
        """创建满足目标时长的音频序列"""
        if not audio_files:
            raise ValueError("没有可用的音频文件")
        
        # 随机打乱音频文件顺序
        random.shuffle(audio_files)
        
        # 计算需要的音频总时长
        current_duration = 0.0
        selected_audios = []
        
        # 选择音频文件直到满足时长要求
        for audio_file in audio_files:
            duration = self.get_audio_duration(audio_file)
            if duration > 0:
                selected_audios.append(audio_file)
                current_duration += duration
                if current_duration >= target_duration:
                    break
        
        if not selected_audios:
            raise ValueError("没有有效的音频文件")
        
        # 如果音频总时长不足，循环使用
        while current_duration < target_duration:
            for audio_file in audio_files:
                duration = self.get_audio_duration(audio_file)
                if duration > 0:
                    selected_audios.append(audio_file)
                    current_duration += duration
                    if current_duration >= target_duration:
                        break
        
        return selected_audios
    
    def _process_single_video(self, video_path: str, audio_sequence: List[str], output_dir: str, 
                             video_volume: float, audio_volume: float, video_index: int, total_videos: int) -> str:
        """处理单个视频文件的音频合并
        
        Args:
            video_path: 视频文件路径
            audio_sequence: 预创建的音频序列
            output_dir: 输出目录
            video_volume: 视频音频音量
            audio_volume: 背景音频音量
            video_index: 当前视频索引
            total_videos: 总视频数量
            
        Returns:
            str: 输出文件路径，失败返回空字符串
        """
        try:
            logger.info(f"处理视频 {video_index+1}/{total_videos}: {os.path.basename(video_path)}")
            
            # 获取视频时长
            video_duration = self.get_video_duration(video_path)
            
            if video_duration <= 0:
                logger.warning(f"无法获取视频时长，跳过: {video_path}")
                return ""
            
            logger.info(f"视频时长: {video_duration:.2f}秒")
            logger.info(f"使用预创建的音频序列，包含 {len(audio_sequence)} 个音频文件")
            
            # 自动生成输出文件名
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_suffix = hashlib.md5(video_path.encode()).hexdigest()[:8]
            output_filename = f"{video_name}_merged_{timestamp}_{hash_suffix}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # 构建ffmpeg命令 - 直接使用混合模式
            self._mix_audio(video_path, audio_sequence, output_path, video_volume, audio_volume, video_duration)
            
            logger.info(f"音频合并完成，输出文件: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"处理视频失败 {video_path}: {e}")
            return ""
    
    def merge_audio_with_video(self, 
                              video_folder: str, 
                              audio_folder: str, 
                              output_dir: str,
                              video_volume: float = 1.0,
                              audio_volume: float = 1.0,
                              max_workers: int = 4) -> List[str]:
        """
        合并视频与随机音频，处理所有视频文件（多线程）
        
        Args:
            video_folder: 视频文件夹路径
            audio_folder: 音频文件夹路径
            output_dir: 输出目录路径
            video_volume: 视频音频音量 (0.0-2.0)
            audio_volume: 背景音频音量 (0.0-2.0)
            max_workers: 最大线程数，默认为4
            
        Returns:
            List[str]: 成功处理的输出文件路径列表
        """
        output_files = []
        
        try:
            # 获取视频和音频文件
            video_files = self.get_video_files(video_folder)
            audio_files = self.get_audio_files(audio_folder)
            
            if not video_files:
                logger.warning("视频文件夹中没有找到视频文件")
                return output_files
            
            if not audio_files:
                logger.warning("音频文件夹中没有找到音频文件")
                return output_files
            
            logger.info(f"找到 {len(video_files)} 个视频文件，开始多线程处理...")
            logger.info(f"使用 {max_workers} 个线程并行处理")
            
            # 遍历所有视频文件，为每个视频创建音频序列
            video_audio_pairs = []
            for i, video_path in enumerate(video_files):
                try:
                    logger.info(f"准备视频 {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
                    
                    # 获取视频时长
                    video_duration = self.get_video_duration(video_path)
                    
                    if video_duration <= 0:
                        logger.warning(f"无法获取视频时长，跳过: {video_path}")
                        continue
                    
                    logger.info(f"视频时长: {video_duration:.2f}秒")
                    
                    # 创建音频序列
                    audio_sequence = self.create_audio_sequence(audio_files, video_duration)
                    logger.info(f"选择了 {len(audio_sequence)} 个音频文件")
                    
                    # 保存视频路径和对应的音频序列
                    video_audio_pairs.append((video_path, audio_sequence, i, len(video_files)))
                    
                except Exception as e:
                    logger.error(f"准备视频失败 {video_path}: {e}")
                    continue
            
            # 使用线程池并行处理视频文件
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，一个视频一个线程
                future_to_video = {
                    executor.submit(
                        self._process_single_video, 
                        video_path, 
                        audio_sequence, 
                        output_dir, 
                        video_volume, 
                        audio_volume, 
                        video_index, 
                        total_videos
                    ): (video_path, video_index) 
                    for video_path, audio_sequence, video_index, total_videos in video_audio_pairs
                }
                
                # 收集结果
                completed_count = 0
                for future in as_completed(future_to_video):
                    video_path, video_index = future_to_video[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        if result:  # 如果返回了输出文件路径
                            output_files.append(result)
                        logger.info(f"进度: {completed_count}/{len(video_audio_pairs)} 完成")
                    except Exception as e:
                        logger.error(f"线程处理异常 {video_path}: {e}")
            
            logger.info(f"多线程处理完成，成功处理 {len(output_files)} 个视频")
            return output_files
            
        except Exception as e:
            logger.error(f"音频合并失败: {e}")
            return output_files
    
    
    def _mix_audio(self, video_path: str, audio_sequence: List[str], output_path: str, video_volume: float, audio_volume: float, target_duration: float):
        """混合音频模式 - 将视频音频与背景音频混合"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_bg_audio = temp_file.name
        
        try:
            # 合并背景音频
            self._concatenate_audios(audio_sequence, temp_bg_audio, target_duration, audio_volume)
            
            # 混音
            video_input = ffmpeg.input(video_path)
            bg_audio_input = ffmpeg.input(temp_bg_audio)
            
            # 调整音量
            video_audio = ffmpeg.filter(video_input['a'], 'volume', video_volume)
            bg_audio = ffmpeg.filter(bg_audio_input['a'], 'volume', audio_volume)
            
            # 混音
            mixed_audio = ffmpeg.filter([video_audio, bg_audio], 'amix', inputs=2, duration='longest')
            
            (
                ffmpeg
                .output(video_input['v'], mixed_audio, output_path, vcodec='copy', acodec='aac')
                .overwrite_output()
                .run()
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_bg_audio):
                os.remove(temp_bg_audio)
    
    def _concatenate_audios(self, audio_files: List[str], output_path: str, target_duration: float, volume: float):
        """连接音频文件"""

        
        if len(audio_files) == 1:
            # 单个音频文件，直接复制并调整音量
            audio_input = ffmpeg.input(audio_files[0])
            audio_filtered = ffmpeg.filter(audio_input['a'], 'volume', volume)
            
            # 如果音频时长超过目标时长，截取
            if self.get_audio_duration(audio_files[0]) > target_duration:
                audio_filtered = ffmpeg.filter(audio_filtered, 'atrim', duration=target_duration)
            
            (
                ffmpeg
                .output(audio_filtered, output_path, acodec='pcm_s16le')
                .overwrite_output()
                .run()
            )
        else:
            # 多个音频文件，需要连接
            # 连接所有音频文件（包括重复的）
            inputs = []
            for audio_file in audio_files:
                audio_input = ffmpeg.input(audio_file)
                inputs.append(audio_input['a'])
            
            # 连接音频
            concatenated = ffmpeg.filter(inputs, 'concat', n=len(inputs), v=0, a=1)
            
            # 统一调整音量
            audio_with_volume = ffmpeg.filter(concatenated, 'volume', volume)
            
            # 截取到目标时长
            final_audio = ffmpeg.filter(audio_with_volume, 'atrim', duration=target_duration)
            
            (
                ffmpeg
                .output(final_audio, output_path, acodec='pcm_s16le')
                .overwrite_output()
                .run()
            )
        return output_path

# ComfyUI节点类
class AudioMergerNode:
    """ComfyUI音频合并节点"""
    
    def __init__(self):
        self.merger = AudioMerger()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "视频文件夹路径"
                }),
                "audio_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "背景音频文件夹路径"
                }),
                "video_prefix": ("STRING", {
                    "default": "video_bgm",
                    "multiline": False,
                    "tooltip": "视频前缀，用于生成输出文件夹名称"
                }),
                "video_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "视频音频音量 (0.0-2.0，设为0则静音视频音频)"
                }),
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "背景音频音量 (0.0-2.0，设为0则静音背景音频)"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "merge_audio"
    CATEGORY = "video_editing/audio"
    
    def merge_audio(self, video_folder, audio_folder, video_prefix, video_volume, audio_volume, max_workers = max(8, max(1, os.cpu_count() - 2 or 1))):
        """执行音频合并"""
        try:
            # 验证输入
            if not video_folder or not audio_folder:
                return ("",)
            
            # 解析路径，支持相对和绝对路径
            resolved_video_folder = resolve_path(video_folder)
            resolved_audio_folder = resolve_path(audio_folder)
            
            if not os.path.exists(resolved_video_folder):
                logger.error(f"视频文件夹不存在: {resolved_video_folder}")
                return ("",)
            
            if not os.path.exists(resolved_audio_folder):
                logger.error(f"音频文件夹不存在: {resolved_audio_folder}")
                return ("",)
            
            # 构建ComfyUI输出目录路径
            output_base_dir = folder_paths.get_output_directory()
            
            # 生成唯一的文件夹名称
            folder_name = generate_unique_folder_name(video_prefix, output_base_dir)
            
            # 创建保存目录
            save_dir = os.path.join(output_base_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 检查是否有视频文件
            video_files = self.merger.get_video_files(resolved_video_folder)
            if not video_files:
                return ("",)
            
            # 检查是否有音频文件
            audio_files = self.merger.get_audio_files(resolved_audio_folder)
            if not audio_files:
                return ("",)
            
            logger.info(f"找到 {len(video_files)} 个视频文件")
            logger.info(f"找到 {len(audio_files)} 个音频文件")
            logger.info(f"输出路径: {save_dir}")
            
            # 执行合并
            output_files = self.merger.merge_audio_with_video(
                video_folder=resolved_video_folder,
                audio_folder=resolved_audio_folder,
                output_dir=save_dir,
                video_volume=video_volume,
                audio_volume=audio_volume,
                max_workers=max_workers
            )
            
            if output_files:
                # 返回保存路径（目录）
                return (save_dir,)
            else:
                return ("",)
                
        except Exception as e:
            logger.error(f"音频合并异常: {str(e)}")
            return ("",)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "AudioMergerNode": AudioMergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioMergerNode": "批量视频添加bgm"
}