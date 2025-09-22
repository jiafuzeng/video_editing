import ffmpeg
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from pathlib import Path
import tempfile
import json
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import folder_paths
import logging
from .utils import generate_unique_folder_name, resolve_path

def get_video_duration(video_path):
    """
    获取视频时长（秒）
    """
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        logging.error(f"获取视频时长失败 {video_path}: {e}")
        return 0


def find_json_file_for_video(video_path):
    """
    为视频文件查找对应的JSON配置文件
    返回: JSON文件路径 或 None 如果不存在
    """
    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem  # 不包含扩展名的文件名
    
    # 尝试不同的JSON文件名
    json_candidates = [
        video_dir / f"{video_stem}.json",
        video_dir / f"{video_stem}.JSON",
    ]
    
    for json_path in json_candidates:
        if json_path.exists():
            return str(json_path)
    
    return None


def load_json_config(json_file_path):
    """
    加载JSON配置文件，提取merged的segments信息
    返回: merged_segments列表 或 None 如果不存在或无效
    """
    try:
        if not os.path.exists(json_file_path):
            logging.error(f"JSON文件不存在: {json_file_path}")
            return None
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查是否有merged配置
        if 'merged' in config and isinstance(config['merged'], dict):
            merged = config['merged']
            segments = merged.get('segments', [])
            
            if segments and isinstance(segments, list):
                # 验证segments格式
                valid_segments = []
                for segment in segments:
                    if (isinstance(segment, dict) and 
                        'start' in segment and 'end' in segment and
                        isinstance(segment['start'], (int, float)) and 
                        isinstance(segment['end'], (int, float)) and
                        segment['start'] >= 0 and segment['end'] > segment['start']):
                        valid_segments.append({
                            'start': float(segment['start']),
                            'end': float(segment['end']),
                            'score': segment.get('score', 0)
                        })
                
                if valid_segments:
                    logging.info(f"JSON配置加载成功: 找到 {len(valid_segments)} 个有效segments")
                    return valid_segments
                else:
                    logging.error("JSON配置中merged.segments格式无效")
                    return None
            else:
                logging.error("JSON配置中merged.segments为空或格式错误")
                return None
        else:
            logging.error("JSON配置中未找到merged配置")
            return None
            
    except Exception as e:
        logging.error(f"加载JSON配置文件失败: {e}")
        return None


def cut_single_segment_without_end(video_path, start_time, end_time, output_path):
    """
    按给定起止时间切分单段视频，不添加结尾视频
    """
    try:
        # 获取主视频信息
        main_probe = ffmpeg.probe(video_path)
        
        # 获取主视频的分辨率
        main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
        main_width = int(main_video_stream['width'])
        main_height = int(main_video_stream['height'])
        
        # 切分主视频并重新编码以确保兼容性
        # 添加容错参数来处理损坏的视频数据
        input_stream = ffmpeg.input(video_path, ss=start_time, t=end_time-start_time, 
                                   **{'fflags': '+ignidx+igndts'})  # 忽略损坏的数据
        # 使用setsar filter来统一SAR参数
        video_stream = input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
        audio_stream = input_stream.audio
        
        (
            ffmpeg
            .output(
                video_stream,
                audio_stream,
                output_path,
                vcodec='libx264',
                preset='fast',
                **{'profile:v': 'main'},
                r=30,  # 固定帧率为30fps
                acodec='aac',
                ar=44100,
                ac=2,
                **{'fflags': '+ignidx+igndts'}  # 输出时也忽略错误
            )
            .overwrite_output()
            .run(quiet=True)
        )
        
        # 验证最终视频时长
        final_probe = ffmpeg.probe(output_path)
        final_duration = float(final_probe['streams'][0]['duration'])
        expected_duration = end_time - start_time
        
        logging.info(f"切分时长: {expected_duration:.2f}s, 实际时长: {final_duration:.2f}s")
        
        if abs(final_duration - expected_duration) > 0.1:  # 允许0.1秒误差
            logging.warning(f"时长不匹配! 期望: {expected_duration:.2f}s, 实际: {final_duration:.2f}s")
        
        return True
    except Exception as e:
        logging.error(f"处理视频失败 {video_path}: {e}")
        # 如果是ffmpeg错误，显示更详细的信息
        if hasattr(e, 'stderr') and e.stderr:
            try:
                error_msg = e.stderr.decode('utf8')
                logging.error(f"FFmpeg错误详情: {error_msg}")
            except:
                pass
        return False


def cut_single_segment_with_end(video_path, start_time, end_time, output_path, prepared_end_path, end_duration):
    """
    按给定起止时间切分单段视频并添加结尾视频，确保音视频同步
    """
    try:
        # 获取主视频信息
        main_probe = ffmpeg.probe(video_path)
        
        # 获取主视频的分辨率
        main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
        main_width = int(main_video_stream['width'])
        main_height = int(main_video_stream['height'])
        
        # 使用临时文件来避免concat的复杂性
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_main = os.path.join(temp_dir, "temp_main.mp4")
            
            # 切分主视频并重新编码以确保兼容性
            # 添加容错参数来处理损坏的视频数据
            input_stream = ffmpeg.input(video_path, ss=start_time, t=end_time-start_time, 
                                       **{'fflags': '+ignidx+igndts'})  # 忽略损坏的数据
            # 使用setsar filter来统一SAR参数
            video_stream = input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            audio_stream = input_stream.audio
            
            (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    temp_main,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,  # 固定帧率为30fps
                    acodec='aac',
                    ar=44100,
                    ac=2,
                    **{'fflags': '+ignidx+igndts'}  # 输出时也忽略错误
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 使用filter_complex进行更可靠的合并
            main_input = ffmpeg.input(temp_main)
            end_input = ffmpeg.input(prepared_end_path)
            
            (
                ffmpeg
                .filter([main_input.video, main_input.audio, end_input.video, end_input.audio], 
                       'concat', n=2, v=1, a=1)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
        
        # 验证最终视频时长
        final_probe = ffmpeg.probe(output_path)
        final_duration = float(final_probe['streams'][0]['duration'])
        expected_duration = (end_time - start_time) + end_duration
        
        logging.info(f"切分时长: {end_time - start_time:.2f}s, 结尾时长: {end_duration:.2f}s, 最终时长: {final_duration:.2f}s")
        
        if abs(final_duration - expected_duration) > 0.1:  # 允许0.1秒误差
            logging.warning(f"时长不匹配! 期望: {expected_duration:.2f}s, 实际: {final_duration:.2f}s")
        
        return True
    except Exception as e:
        logging.error(f"处理视频失败 {video_path}: {e}")
        # 如果是ffmpeg错误，显示更详细的信息
        if hasattr(e, 'stderr') and e.stderr:
            try:
                error_msg = e.stderr.decode('utf8')
                logging.error(f"FFmpeg错误详情: {error_msg}")
            except:
                pass
        return False




def cut_video_without_end(video_path, cut_duration, video_output_dir, merged_segments=None):
    """
    将视频按 cut_duration 切分为多段，不添加结尾视频，输出到 video_output_dir
    """
    video_name = Path(video_path).stem
    tid = threading.get_ident()
    video_duration = get_video_duration(video_path)

    # 原有的处理逻辑
    if video_duration < cut_duration:
        logging.info(f"[TID {tid}] 跳过视频 {video_name}: 时长 {video_duration:.2f}s 小于切分时长 {cut_duration}s")
        return

    # 计算可以切分的段数，确保不超过视频时长
    num_segments = int(video_duration // cut_duration)

    # 确保最后一段不会超出视频时长
    if num_segments * cut_duration >= video_duration:
        num_segments = max(1, num_segments - 1)

    logging.info(f"[TID {tid}] 处理视频 {video_name}: 总时长 {video_duration:.2f}s, 将切分为 {num_segments} 段")
    
    # 切分视频
    cut_segment_without_end(video_path, cut_duration, video_output_dir, num_segments, tid)


def cut_video_with_end(video_path, cut_duration, end_video_path, video_output_dir, merged_segments=None):
    """
    将视频按 cut_duration 切分为多段，并为每段添加结尾视频，输出到 video_output_dir
    如果提供了merged_segments，先按segments提取视频片段，再对每个片段进行切分
    """
    video_name = Path(video_path).stem
    tid = threading.get_ident()
    video_duration = get_video_duration(video_path)

    # 原有的处理逻辑
    if video_duration < cut_duration:
        logging.info(f"[TID {tid}] 跳过视频 {video_name}: 时长 {video_duration:.2f}s 小于切分时长 {cut_duration}s")
        return

    # 计算可以切分的段数，确保不超过视频时长
    num_segments = int(video_duration // cut_duration)

    # 确保最后一段不会超出视频时长
    if num_segments * cut_duration >= video_duration:
        num_segments = max(1, num_segments - 1)

    logging.info(f"[TID {tid}] 处理视频 {video_name}: 总时长 {video_duration:.2f}s, 将切分为 {num_segments} 段")
    
    # 切分视频
    cut_segment_with_end(video_path, cut_duration, end_video_path, video_output_dir, num_segments, tid)


def cut_segment_without_end(video_path, cut_duration, output_dir, num_segments, tid):
    """
    对单个视频片段进行切分，不添加结尾视频
    """
    video_duration = get_video_duration(video_path)
    video_name = Path(video_path).stem
    
    # 切分视频
    for i in range(num_segments):
        start_time = i * cut_duration
        end_time = (i + 1) * cut_duration

        # 确保切分时间不超过视频时长
        if end_time > video_duration:
            end_time = video_duration

        # 如果切分时长太短，跳过
        if end_time - start_time < cut_duration * 0.5:  # 如果切分时长小于一半，跳过
            logging.info(f"[TID {tid}] 跳过: segment_{i+1:03d}.mp4 (切分时长太短: {end_time - start_time:.2f}s)")
            continue

        # 生成唯一文件名，避免多线程冲突
        output_filename = f"{video_name}_segment_{i+1:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        success = cut_single_segment_without_end(video_path, start_time, end_time, output_path)
        if success:
            logging.info(f"[TID {tid}] 完成: {output_filename}")
        else:
            logging.error(f"[TID {tid}] 失败: {output_filename}")


def cut_segment_with_end(video_path, cut_duration, end_video_path, output_dir, num_segments, tid):
    """
    对单个视频片段进行切分并添加结尾视频
    """
    video_duration = get_video_duration(video_path)
    video_name = Path(video_path).stem
    
    # 预处理结尾视频（循环外一次），按主视频分辨率/参数
    try:
        main_probe = ffmpeg.probe(video_path)
        main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
        main_width = int(main_video_stream['width'])
        main_height = int(main_video_stream['height'])
        end_probe = ffmpeg.probe(end_video_path)
        end_duration = float(end_probe['streams'][0]['duration'])
    except Exception as e:
        logging.error(f"[TID {tid}] 准备结尾视频失败: {e}")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        prepared_end_path = os.path.join(temp_dir, "prepared_end.mp4")

        try:
            end_input_stream = ffmpeg.input(end_video_path)
            end_video_stream = end_input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            end_audio_stream = end_input_stream.audio

            (
                ffmpeg
                .output(
                    end_video_stream,
                    end_audio_stream,
                    prepared_end_path,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logging.error(f"[TID {tid}] 结尾视频转码失败: {e}")
            return

        # 切分视频并添加结尾
        for i in range(num_segments):
            start_time = i * cut_duration
            end_time = (i + 1) * cut_duration

            # 确保切分时间不超过视频时长
            if end_time > video_duration:
                end_time = video_duration

            # 如果切分时长太短，跳过
            if end_time - start_time < cut_duration * 0.5:  # 如果切分时长小于一半，跳过
                logging.info(f"[TID {tid}] 跳过: segment_{i+1:03d}.mp4 (切分时长太短: {end_time - start_time:.2f}s)")
                continue

            # 生成唯一文件名，避免多线程冲突
            output_filename = f"{video_name}_segment_{i+1:03d}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            success = cut_single_segment_with_end(video_path, start_time, end_time, output_path, prepared_end_path, end_duration)
            if success:
                logging.info(f"[TID {tid}] 完成: {output_filename}")
            else:
                logging.error(f"[TID {tid}] 失败: {output_filename}")


def process_video(video_path, cut_duration, end_video_path, output_dir, global_merged_segments=None):
    """
    处理单个视频文件，自动检测对应的JSON配置文件
    """
    video_name = Path(video_path).stem
    tid = threading.get_ident()
    
    # 直接使用输出目录，不创建子目录
    video_output_dir = output_dir
    
    # 处理JSON配置
    merged_segments = global_merged_segments
    
    if merged_segments is None:
        # 自动检测JSON配置文件
        json_file_path = find_json_file_for_video(video_path)
        if json_file_path:
            logging.info(f"[TID {tid}] 找到JSON配置文件: {json_file_path}")
            merged_segments = load_json_config(json_file_path)
            if merged_segments is None:
                logging.warning(f"[TID {tid}] JSON配置加载失败，将使用常规处理模式")
        else:
            logging.info(f"[TID {tid}] 未找到JSON配置文件，使用常规处理模式")
    else:
        logging.info(f"[TID {tid}] 使用全局JSON配置")
    
    # 处理视频
    logging.info(f"[TID {tid}] 开始处理: {video_name}")
    if end_video_path:
        cut_video_with_end(video_path, cut_duration, end_video_path, video_output_dir, merged_segments)
    else:
        cut_video_without_end(video_path, cut_duration, video_output_dir, merged_segments)
    logging.info(f"[TID {tid}] 处理完成: {video_name}")


def process_videos_folder(input_folder, cut_duration, end_video_path, output_dir, max_workers=4, global_merged_segments=None):
    """
    处理文件夹中的所有视频，每个视频自动检测对应的JSON配置文件
    """
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.m4v']
    
    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
        video_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not video_files:
        logging.warning(f"在文件夹 {input_folder} 中未找到视频文件")
        return
    
    logging.info(f"找到 {len(video_files)} 个视频文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用线程池处理视频
    tid_main = threading.get_ident()
    logging.info(f"[TID {tid_main}] 准备启动线程池，workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_file in video_files:
            logging.info(f"[TID {tid_main}] 提交任务: {Path(video_file).name}")
            future = executor.submit(process_video, video_file, cut_duration, end_video_path, output_dir, global_merged_segments)
            future_to_video[future] = video_file
        
        # 等待所有任务完成
        for future in as_completed(future_to_video):
            video_file = future_to_video[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"[TID {tid_main}] 处理视频时发生错误: {Path(video_file).name}: {e}")


class VideoCutNode(ComfyNodeABC):
    """视频切分节点 - 将视频按指定时长切分"""
    
    DESCRIPTION = "将视频按指定时长切分，支持JSON配置文件自动检测"
    CATEGORY = "video/video_editing"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "input_folder": (IO.STRING, {"default": "", "tooltip": "输入文件夹路径（支持相对路径和绝对路径）"}),
                "cut_duration": (IO.FLOAT, {"default": 9.5, "min": 1.0, "max": 3600.0, "step": 0.1, "tooltip": "每个切分视频的时长（秒）"}),
                "output_folder_prefix": (IO.STRING, {"default": "video_cut", "tooltip": "输出文件夹前缀"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    
    def execute(self, input_folder: str, cut_duration: float, output_folder_prefix: str):
        """执行视频切分处理"""
        try:
            # 解析输入路径（支持相对路径和绝对路径）
            input_folder_path = resolve_path(input_folder)
            logging.info(f"使用输入路径: {input_folder_path}")
            
            # 验证输入路径是否存在
            if not os.path.exists(input_folder_path):
                logging.error(f"输入文件夹不存在: {input_folder_path}")
                return ("",)
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            unique_folder_name = generate_unique_folder_name(output_folder_prefix, output_dir)
            batch_output_dir = os.path.join(output_dir, unique_folder_name)
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # 处理文件夹中的所有视频
            logging.info(f"开始处理文件夹: {input_folder}")
            logging.info(f"切分时长: {cut_duration} 秒")
            logging.info(f"输出目录: {batch_output_dir}")
            logging.info(f"生成的唯一文件夹名: {unique_folder_name}")
            logging.info(f"线程数: 10")
            
            # 执行批量视频处理
            process_videos_folder(input_folder_path, cut_duration, None, batch_output_dir, 10, None)
            
            return (batch_output_dir,)
            
        except ValueError as e:
            # 输入验证错误
            logging.error(f"输入错误: {str(e)}")
            return ("",)
        except Exception as e:
            # 其他错误
            logging.error(f"处理视频时发生错误: {str(e)}")
            return ("",)




class VideoAddNode(ComfyNodeABC):
    """视频添加节点 - 为文件夹中的所有视频添加指定视频（前或后）"""
    
    DESCRIPTION = "为文件夹中的所有视频添加指定视频，支持在前或后添加"
    CATEGORY = "video/video_editing"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "input_folder": (IO.STRING, {"default": "", "tooltip": "输入文件夹路径（支持相对路径和绝对路径）"}),
                "add_video": (IO.STRING, {"default": "", "tooltip": "要添加的视频文件路径（支持相对路径和绝对路径）"}),
                "add_position": (["before", "after"], {"default": "after", "tooltip": "添加位置：before=在主视频前添加，after=在主视频后添加"}),
                "output_folder_prefix": (IO.STRING, {"default": "video_add", "tooltip": "输出文件夹前缀"}),
            }
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    
    def execute(self, input_folder: str, add_video: str, add_position: str, output_folder_prefix: str):
        """执行视频添加处理"""
        try:
            # 解析输入路径
            input_folder_path = resolve_path(input_folder)
            
            # 获取要添加的视频路径
            if not add_video or add_video.strip() == "":
                return ("",)
            
            add_video_path = resolve_path(add_video.strip())
            if not os.path.exists(add_video_path):
                return ("",)
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            unique_folder_name = generate_unique_folder_name(output_folder_prefix, output_dir)
            batch_output_dir = os.path.join(output_dir, unique_folder_name)
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # 处理文件夹中的所有视频
            logging.info(f"开始处理文件夹: {input_folder}")
            logging.info(f"要添加的视频: {add_video}")
            logging.info(f"添加位置: {'前' if add_position == 'before' else '后'}")
            logging.info(f"输出目录: {batch_output_dir}")
            logging.info(f"线程数: 10")
            
            # 根据添加位置选择处理函数
            if add_position == "before":
                # 在主视频前添加
                prepend_start_to_videos_folder(input_folder_path, add_video_path, batch_output_dir, 10)
            else:
                # 在主视频后添加
                append_end_to_videos_folder(input_folder_path, add_video_path, batch_output_dir, 10)
            
            return (batch_output_dir,)
            
        except ValueError as e:
            # 输入验证错误
            return ("",)
        except Exception as e:
            # 其他错误
            error_msg = f"处理视频时发生错误: {str(e)}"
            logging.error(error_msg)
            return ("",)


def append_end_to_videos_folder(input_folder, end_video_path, output_dir, max_workers=4):
    """
    为文件夹中的所有视频添加结尾视频，保持目录结构
    """
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    # 递归获取所有视频文件，保持相对路径
    video_files = []
    input_folder_path = Path(input_folder)
    
    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                file_path = os.path.join(root, file)
                # 确保是文件而不是目录
                if os.path.isfile(file_path):
                    # 计算相对于输入文件夹的路径
                    relative_path = root_path.relative_to(input_folder_path)
                    video_files.append((file_path, relative_path))
    
    if not video_files:
        logging.warning(f"在文件夹 {input_folder} 中未找到视频文件")
        return
    
    logging.info(f"找到 {len(video_files)} 个视频文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取结尾视频文件夹中的所有视频文件
    end_video_files = []
    for file in os.listdir(end_video_path):
        file_path = os.path.join(end_video_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']):
            end_video_files.append(file_path)
    
    if not end_video_files:
        logging.error(f"结尾视频文件夹中没有找到视频文件: {end_video_path}")
        return
    
    logging.info(f"找到 {len(end_video_files)} 个结尾视频文件")
    
    # 使用线程池处理视频
    tid_main = threading.get_ident()
    logging.info(f"[TID {tid_main}] 准备启动线程池，workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for i, (video_file, relative_path) in enumerate(video_files):
            # 循环选择结尾视频
            end_video_index = i % len(end_video_files)
            selected_end_video = end_video_files[end_video_index]
            
            logging.info(f"[TID {tid_main}] 提交任务: {relative_path / Path(video_file).name} -> 使用结尾视频: {os.path.basename(selected_end_video)}")
            
            future = executor.submit(append_end_to_single_video, video_file, selected_end_video, output_dir)
            future_to_video[future] = (video_file, relative_path)
        
        # 等待所有任务完成
        for future in as_completed(future_to_video):
            video_file, relative_path = future_to_video[future]
            try:
                future.result()
                logging.info(f"[TID {tid_main}] 完成: {relative_path / Path(video_file).name}")
            except Exception as e:
                logging.error(f"[TID {tid_main}] 处理视频时发生错误: {relative_path / Path(video_file).name}: {e}")
    
    logging.info(f"批量添加结尾视频完成！输出目录: {output_dir}")


def append_end_to_single_video(video_path, end_video_path, output_dir):
    """
    为单个视频添加结尾视频
    """
    video_name = Path(video_path).stem
    tid = threading.get_ident()
    
    try:
        # 获取主视频信息
        main_probe = ffmpeg.probe(video_path)
        main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
        main_width = int(main_video_stream['width'])
        main_height = int(main_video_stream['height'])
        
        # 获取结尾视频信息
        end_probe = ffmpeg.probe(end_video_path)
        end_duration = float(end_probe['streams'][0]['duration'])
        
        # 输出文件路径
        output_filename = f"{video_name}_with_end.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 使用临时文件来避免concat的复杂性
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_main = os.path.join(temp_dir, "temp_main.mp4")
            prepared_end_path = os.path.join(temp_dir, "prepared_end.mp4")
            
            # 处理主视频
            main_input_stream = ffmpeg.input(video_path, **{'fflags': '+ignidx+igndts'})
            main_video_stream = main_input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            main_audio_stream = main_input_stream.audio
            
            (
                ffmpeg
                .output(
                    main_video_stream,
                    main_audio_stream,
                    temp_main,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2,
                    **{'fflags': '+ignidx+igndts'}
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 处理结尾视频
            end_input_stream = ffmpeg.input(end_video_path)
            end_video_stream = end_input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            end_audio_stream = end_input_stream.audio
            
            (
                ffmpeg
                .output(
                    end_video_stream,
                    end_audio_stream,
                    prepared_end_path,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 合并主视频和结尾视频
            main_input = ffmpeg.input(temp_main)
            end_input = ffmpeg.input(prepared_end_path)
            
            (
                ffmpeg
                .filter([main_input.video, main_input.audio, end_input.video, end_input.audio], 
                       'concat', n=2, v=1, a=1)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
        
        # 验证最终视频时长
        final_probe = ffmpeg.probe(output_path)
        final_duration = float(final_probe['streams'][0]['duration'])
        main_duration = float(main_probe['streams'][0]['duration'])
        expected_duration = main_duration + end_duration
        
        logging.info(f"[TID {tid}] 完成: {output_filename}")
        logging.info(f"[TID {tid}] 主视频时长: {main_duration:.2f}s, 结尾时长: {end_duration:.2f}s, 最终时长: {final_duration:.2f}s")
        
        if abs(final_duration - expected_duration) > 0.1:  # 允许0.1秒误差
            logging.warning(f"[TID {tid}] 时长不匹配! 期望: {expected_duration:.2f}s, 实际: {final_duration:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"[TID {tid}] 处理视频失败 {video_path}: {e}")
        # 如果是ffmpeg错误，显示更详细的信息
        if hasattr(e, 'stderr') and e.stderr:
            try:
                error_msg = e.stderr.decode('utf8')
                logging.error(f"[TID {tid}] FFmpeg错误详情: {error_msg}")
            except:
                pass
        return False


def prepend_start_to_videos_folder(input_folder, start_video_path, output_dir, max_workers=4):
    """
    为文件夹中的所有视频添加开始视频，保持目录结构
    """
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    # 递归获取所有视频文件，保持相对路径
    video_files = []
    input_folder_path = Path(input_folder)
    
    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                file_path = os.path.join(root, file)
                # 确保是文件而不是目录
                if os.path.isfile(file_path):
                    # 计算相对于输入文件夹的路径
                    relative_path = root_path.relative_to(input_folder_path)
                    video_files.append((file_path, relative_path))
    
    if not video_files:
        logging.warning(f"在文件夹 {input_folder} 中未找到视频文件")
        return
    
    logging.info(f"找到 {len(video_files)} 个视频文件")

    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取开始视频文件夹中的所有视频文件
    start_video_files = []
    for file in os.listdir(start_video_path):
        file_path = os.path.join(start_video_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']):
            start_video_files.append(file_path)
    
    if not start_video_files:
        logging.error(f"开始视频文件夹中没有找到视频文件: {start_video_path}")
        return
    
    logging.info(f"找到 {len(start_video_files)} 个开始视频文件")
    
    # 使用线程池处理视频
    tid_main = threading.get_ident()
    logging.info(f"[TID {tid_main}] 准备启动线程池，workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for i, (video_file, relative_path) in enumerate(video_files):
            # 循环选择开始视频
            start_video_index = i % len(start_video_files)
            selected_start_video = start_video_files[start_video_index]
            
            logging.info(f"[TID {tid_main}] 提交任务: {relative_path / Path(video_file).name} -> 使用开始视频: {os.path.basename(selected_start_video)}")
            
            future = executor.submit(prepend_start_to_single_video, video_file, selected_start_video, output_dir)
            future_to_video[future] = (video_file, relative_path)
        
        # 等待所有任务完成
        for future in as_completed(future_to_video):
            video_file, relative_path = future_to_video[future]
            try:
                future.result()
                logging.info(f"[TID {tid_main}] 完成: {relative_path / Path(video_file).name}")
            except Exception as e:
                logging.error(f"[TID {tid_main}] 处理视频时发生错误: {relative_path / Path(video_file).name}: {e}")
    
    logging.info(f"批量添加开始视频完成！输出目录: {output_dir}")


def prepend_start_to_single_video(video_path, start_video_path, output_dir):
    """
    为单个视频添加开始视频
    """
    video_name = Path(video_path).stem
    tid = threading.get_ident()
    
    try:
        # 检查文件是否存在且不是目录
        if not os.path.isfile(video_path):
            logging.warning(f"[TID {tid}] 跳过非文件路径: {video_path}")
            return False
            
        # 获取主视频信息
        try:
            main_probe = ffmpeg.probe(video_path)
            main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
            main_width = int(main_video_stream['width'])
            main_height = int(main_video_stream['height'])
        except Exception as e:
            logging.error(f"[TID {tid}] 无法解析主视频 {video_path}: {e}")
            return False
        
        #
        # 获取开始视频信息
        try:
            start_probe = ffmpeg.probe(start_video_path)
            start_duration = float(start_probe['streams'][0]['duration'])
        except Exception as e:
            logging.error(f"[TID {tid}] 无法解析开始视频 {start_video_path}: {e}")
            return False
        
        # 输出文件路径
        output_filename = f"{video_name}_with_start.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 使用临时文件来避免concat的复杂性
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_main = os.path.join(temp_dir, "temp_main.mp4")
            prepared_start_path = os.path.join(temp_dir, "prepared_start.mp4")
            
            # 处理主视频
            main_input_stream = ffmpeg.input(video_path, **{'fflags': '+ignidx+igndts'})
            main_video_stream = main_input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            main_audio_stream = main_input_stream.audio
            
            (
                ffmpeg
                .output(
                    main_video_stream,
                    main_audio_stream,
                    temp_main,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2,
                    **{'fflags': '+ignidx+igndts'}
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 处理开始视频
            start_input_stream = ffmpeg.input(start_video_path)
            start_video_stream = start_input_stream.video.filter('scale', main_width, main_height, flags='lanczos').filter('setsar', '1')
            start_audio_stream = start_input_stream.audio
            
            (
                ffmpeg
                .output(
                    start_video_stream,
                    start_audio_stream,
                    prepared_start_path,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 合并开始视频和主视频（开始视频在前）
            start_input = ffmpeg.input(prepared_start_path)
            main_input = ffmpeg.input(temp_main)
            
            (
                ffmpeg
                .filter([start_input.video, start_input.audio, main_input.video, main_input.audio], 
                       'concat', n=2, v=1, a=1)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
        
        # 验证最终视频时长
        final_probe = ffmpeg.probe(output_path)
        final_duration = float(final_probe['streams'][0]['duration'])
        main_duration = float(main_probe['streams'][0]['duration'])
        expected_duration = start_duration + main_duration
        
        logging.info(f"[TID {tid}] 完成: {output_filename}")
        logging.info(f"[TID {tid}] 开始视频时长: {start_duration:.2f}s, 主视频时长: {main_duration:.2f}s, 最终时长: {final_duration:.2f}s")
        
        if abs(final_duration - expected_duration) > 0.1:  # 允许0.1秒误差
            logging.warning(f"[TID {tid}] 时长不匹配! 期望: {expected_duration:.2f}s, 实际: {final_duration:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"[TID {tid}] 处理视频失败 {video_path}: {e}")
        # 如果是ffmpeg错误，显示更详细的信息
        if hasattr(e, 'stderr') and e.stderr:
            try:
                error_msg = e.stderr.decode('utf8')
                logging.error(f"[TID {tid}] FFmpeg错误详情: {error_msg}")
            except:
                pass
        return False

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "VideoCutNode": VideoCutNode,
    "VideoAddNode": VideoAddNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCutNode": "视频切片-按固定时间",
    "VideoAddNode": "分镜拼接",
}