import os
import random
import ffmpeg
import glob
import tempfile
import datetime
from typing import List


class AudioMerger:
    """音频合并器，支持游戏视频与随机音频的合并"""
    
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
    
    def get_video_files(self, game_folder: str) -> List[str]:
        """获取游戏文件夹中的所有视频文件"""
        video_files = []
        for ext in self.supported_video_formats:
            pattern = os.path.join(game_folder, f"*{ext}")
            video_files.extend(glob.glob(pattern))
            pattern = os.path.join(game_folder, f"*{ext.upper()}")
            video_files.extend(glob.glob(pattern))
        return video_files
    
    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"获取视频时长失败: {e}")
            return 0.0
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"获取音频时长失败: {e}")
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
    
    def merge_audio_with_video(self, 
                              game_folder: str, 
                              audio_folder: str, 
                              output_dir: str,
                              merge_mode: str = "mix",  # "replace", "mix"
                              game_volume: float = 1.0,
                              audio_volume: float = 1.0) -> List[str]:
        """
        合并游戏视频与随机音频，处理所有视频文件
        
        Args:
            game_folder: 游戏视频文件夹路径
            audio_folder: 音频文件夹路径
            output_dir: 输出目录路径
            merge_mode: 合并模式 ("replace": 替换游戏音频, "mix": 混音)
            game_volume: 游戏音频音量 (0.0-2.0)
            audio_volume: 背景音频音量 (0.0-2.0)
            
        Returns:
            List[str]: 成功处理的输出文件路径列表
        """
        output_files = []
        
        try:
            # 获取视频和音频文件
            video_files = self.get_video_files(game_folder)
            audio_files = self.get_audio_files(audio_folder)
            
            if not video_files:
                print("游戏文件夹中没有找到视频文件")
                return output_files
            
            if not audio_files:
                print("音频文件夹中没有找到音频文件")
                return output_files
            
            print(f"找到 {len(video_files)} 个视频文件，开始处理...")
            
            # 遍历所有视频文件
            for i, video_path in enumerate(video_files):
                try:
                    print(f"处理视频 {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
                    
                    # 获取视频时长
                    video_duration = self.get_video_duration(video_path)
                    
                    if video_duration <= 0:
                        print(f"无法获取视频时长，跳过: {video_path}")
                        continue
                    
                    print(f"视频时长: {video_duration:.2f}秒")
                    
                    # 创建音频序列
                    audio_sequence = self.create_audio_sequence(audio_files, video_duration)
                    print(f"选择了 {len(audio_sequence)} 个音频文件")
                    
                    # 自动生成输出文件名
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{video_name}_merged_{timestamp}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 构建ffmpeg命令
                    if merge_mode == "replace":
                        # 完全替换游戏音频
                        self._replace_audio(video_path, audio_sequence, output_path, audio_volume, video_duration)
                    elif merge_mode == "mix":
                        # 混音模式
                        self._mix_audio(video_path, audio_sequence, output_path, game_volume, audio_volume, video_duration)
                    else:
                        print(f"不支持的合并模式: {merge_mode}")
                        continue
                    
                    output_files.append(output_path)
                    print(f"音频合并完成，输出文件: {output_path}")
                    
                except Exception as e:
                    print(f"处理视频失败 {video_path}: {e}")
                    continue
            
            print(f"处理完成，成功处理 {len(output_files)} 个视频")
            return output_files
            
        except Exception as e:
            print(f"音频合并失败: {e}")
            return output_files
    
    def _replace_audio(self, video_path: str, audio_sequence: List[str], output_path: str, audio_volume: float, target_duration: float):
        """替换音频模式 - 完全替换游戏音频"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio = temp_file.name
        
        try:
            # 合并音频文件
            self._concatenate_audios(audio_sequence, temp_audio, target_duration, audio_volume)
            
            # 直接替换：提取视频流 + 新音频流
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(temp_audio)
            
            (
                ffmpeg
                .output(video_input['v'], audio_input['a'], output_path, vcodec='copy', acodec='aac')
                .overwrite_output()
                .run()
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def _mix_audio(self, video_path: str, audio_sequence: List[str], output_path: str, game_volume: float, audio_volume: float, target_duration: float):
        """混音模式"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_bg_audio = temp_file.name
        
        try:
            # 合并背景音频
            self._concatenate_audios(audio_sequence, temp_bg_audio, target_duration, audio_volume)
            
            # 混音
            video_input = ffmpeg.input(video_path)
            bg_audio_input = ffmpeg.input(temp_bg_audio)
            
            # 调整音量
            game_audio = ffmpeg.filter(video_input['a'], 'volume', game_volume)
            bg_audio = ffmpeg.filter(bg_audio_input['a'], 'volume', audio_volume)
            
            # 混音
            mixed_audio = ffmpeg.filter([game_audio, bg_audio], 'amix', inputs=2, duration='longest')
            
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
            inputs = []
            for audio_file in audio_files:
                audio_input = ffmpeg.input(audio_file)
                audio_filtered = ffmpeg.filter(audio_input['a'], 'volume', volume)
                inputs.append(audio_filtered)
            
            # 连接音频
            concatenated = ffmpeg.filter(inputs, 'concat', n=len(inputs), v=0, a=1)
            
            # 截取到目标时长
            final_audio = ffmpeg.filter(concatenated, 'atrim', duration=target_duration)
            
            (
                ffmpeg
                .output(final_audio, output_path, acodec='pcm_s16le')
                .overwrite_output()
                .run()
            )


# ComfyUI节点类
class AudioMergerNode:
    """ComfyUI音频合并节点"""
    
    def __init__(self):
        self.merger = AudioMerger()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "game_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "游戏视频文件夹路径"
                }),
                "audio_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "背景音频文件夹路径"
                }),
                "save_folder": ("STRING", {
                    "default": "audio_merged",
                    "multiline": False,
                    "tooltip": "保存文件夹名称（保存在ComfyUI输出目录下）"
                }),
                "merge_mode": (["replace", "mix"], {
                    "default": "mix",
                    "tooltip": "音频融合模式: replace=完全替换游戏音频, mix=游戏音频+背景音频混音"
                }),
                "game_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "游戏音频音量 (0.0-2.0)"
                }),
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "背景音频音量 (0.0-2.0)"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "merge_audio"
    CATEGORY = "video_editing/audio"
    
    def merge_audio(self, game_folder, audio_folder, save_folder, merge_mode, game_volume, audio_volume):
        """执行音频合并"""
        try:
            # 验证输入
            if not game_folder or not audio_folder:
                return ("",)
            
            if not os.path.exists(game_folder):
                return ("",)
            
            if not os.path.exists(audio_folder):
                return ("",)
            
            # 构建ComfyUI输出目录路径
            try:
                import folder_paths
                output_base_dir = folder_paths.get_output_directory()
            except ImportError:
                # 如果无法获取ComfyUI输出目录，使用当前目录下的output文件夹
                output_base_dir = os.path.join(os.getcwd(), "output")
            
            # 创建保存目录
            save_dir = os.path.join(output_base_dir, save_folder)
            os.makedirs(save_dir, exist_ok=True)
            
            # 检查是否有视频文件
            video_files = self.merger.get_video_files(game_folder)
            if not video_files:
                return ("",)
            
            # 检查是否有音频文件
            audio_files = self.merger.get_audio_files(audio_folder)
            if not audio_files:
                return ("",)
            
            print(f"找到 {len(video_files)} 个视频文件")
            print(f"找到 {len(audio_files)} 个音频文件")
            print(f"输出路径: {save_dir}")
            
            # 执行合并
            output_files = self.merger.merge_audio_with_video(
                game_folder=game_folder,
                audio_folder=audio_folder,
                output_dir=save_dir,
                merge_mode=merge_mode,
                game_volume=game_volume,
                audio_volume=audio_volume
            )
            
            if output_files:
                # 返回保存路径（目录）
                return (save_dir,)
            else:
                return ("",)
                
        except Exception as e:
            print(f"音频合并异常: {str(e)}")
            return ("",)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "AudioMergerNode": AudioMergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioMergerNode": "Audio Merger (音频合并器)"
}


# 使用示例
if __name__ == "__main__":
    # 创建音频合并器实例
    merger = AudioMerger()
    
    # 示例用法
    game_folder = "/path/to/game/videos"
    audio_folder = "/path/to/background/audio"
    output_path = "merged_output.mp4"
    
    # 执行合并
    success = merger.merge_audio_with_video(
        game_folder=game_folder,
        audio_folder=audio_folder,
        output_path=output_path,
        merge_mode="mix",  # 混音模式
        game_volume=0.8,   # 游戏音频音量
        audio_volume=0.6   # 背景音频音量
    )
    
    if success:
        print("音频合并成功！")
    else:
        print("音频合并失败！")