import os
import glob
import tempfile
import logging
from pathlib import Path
import ffmpeg
import folder_paths
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from .utils import generate_unique_folder_name, resolve_path

# 配置logger
logger = logging.getLogger(__name__)

class VideoMergeNode(ComfyNodeABC):
    """
    视频合并节点
    将素材视频和主视频按照指定位置关系合并
    支持上下位置合并，自动处理尺寸和时间轴
    """
    
    DESCRIPTION = "视频合并节点 - 支持相对路径和绝对路径"
    CATEGORY = "video/video_editing"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "upper_video_folder": (IO.STRING, {"default": "", "tooltip": "上方视频文件夹路径（支持相对路径和绝对路径）"}),
                "lower_video_folder": (IO.STRING, {"default": "", "tooltip": "下方视频文件夹路径（支持相对路径和绝对路径）"}),
                "main_video_source": (["upper", "lower"], {"default": "upper", "tooltip": "选择主体视频：upper=上方视频作为主视频, lower=下方视频作为主视频"}),
                "upper_video_volume": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "上方视频音频音量 (0.0-2.0)"}),
                "lower_video_volume": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "下方视频音频音量 (0.0-2.0)"}),
                "output_folder_prefix": (IO.STRING, {"default": "video_merge", "tooltip": "输出文件夹前缀"}),
            },
            "optional": {
                "gif_path": (IO.STRING, {"default": "", "tooltip": "GIF文件夹路径（支持相对路径和绝对路径），GIF文件与主视频依次绑定，不足时循环使用"}),
            }
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "merge_videos"
    OUTPUT_NODE = True
    
    def get_video_info(self, video_path, threshold_db=-60.0):
        """获取视频信息"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                return None
            
            logger.info(f"音频检测 - 文件: {os.path.basename(video_path)}")
            logger.info(f"  静音阈值: {threshold_db} dB")
            
            # 第一步：检查是否有音轨
            has_audio_track = audio_stream is not None
            logger.info(f"  第一步 - 音轨检测: {'有音轨' if has_audio_track else '无音轨'}")
            
            if audio_stream:
                logger.info(f"    音频编码: {audio_stream.get('codec_name', '未知')}")
                logger.info(f"    音频时长: {audio_stream.get('duration', '未知')}")
                logger.info(f"    采样率: {audio_stream.get('sample_rate', '未知')}")
            
            # 第二步：如果有音轨，检测音量
            has_audio = False
            if has_audio_track:
                logger.info(f"  第二步 - 音量检测:")
                try:
                    # 使用ffmpeg-python分析音量
                    input_stream = ffmpeg.input(video_path)
                    audio_stream_test = input_stream.audio
                    
                    # 创建音量检测流
                    volume_stream = audio_stream_test.filter('volumedetect')
                    
                    # 输出到null设备进行分析
                    output_stream = ffmpeg.output(volume_stream, 'pipe:', format='null')
                    
                    # 运行分析
                    process = ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True, quiet=True)
                    
                    # 解析stderr中的音量信息
                    stderr_output = process[1].decode('utf-8') if process[1] else ''
                    
                    if 'mean_volume:' in stderr_output:
                        # 提取音量信息
                        lines = stderr_output.split('\n')
                        for line in lines:
                            if 'mean_volume:' in line:
                                volume_str = line.split('mean_volume:')[1].strip()
                                try:
                                    volume_db = float(volume_str.split()[0])
                                    logger.info(f"    平均音量: {volume_db} dB")
                                    
                                    # 如果音量大于阈值，认为有声音
                                    has_audio = volume_db > threshold_db
                                    logger.info(f"    音量判断: {'有声音' if has_audio else '静音'} (阈值: {threshold_db} dB)")
                                    break
                                except Exception as parse_e:
                                    logger.warning(f"    音量解析失败: {parse_e}")
                                    has_audio = True  # 解析失败时默认认为有声音
                                    logger.info(f"    音量判断: 有声音（解析失败）")
                                    break
                        else:
                            has_audio = True
                            logger.info(f"    音量判断: 有声音（未找到音量信息）")
                    else:
                        has_audio = True
                        logger.info(f"    音量判断: 有声音（无音量信息）")
                        
                except Exception as volume_e:
                    logger.warning(f"    音量检测异常: {volume_e}")
                    has_audio = True  # 检测失败时默认认为有声音
                    logger.info(f"    音量判断: 有声音（检测失败）")
            else:
                logger.info(f"  第二步 - 跳过音量检测（无音轨）")
                has_audio = False
            
            logger.info(f"  最终判断: {'有音频' if has_audio else '无音频'}")
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'duration': float(probe['format']['duration']),
                'fps': eval(video_stream['r_frame_rate']),
                'has_audio': has_audio
            }
        except Exception as e:
            logger.error(f"获取视频信息失败 {video_path}: {str(e)}")
            return None
    
    def resize_video_to_width(self, input_path, target_width, output_path):
        """将视频缩放到指定宽度，保持宽高比"""
        try:
            video_info = self.get_video_info(input_path)
            if not video_info:
                return False
                
            original_width = video_info['width']
            original_height = video_info['height']
            
            # 计算新的高度
            new_height = int((target_width * original_height) / original_width)
            
            # 确保高度是偶数（视频编码要求）
            if new_height % 2 != 0:
                new_height += 1
            
            # 使用ffmpeg进行缩放，兼容有声音和没有声音的情况
            input_stream = ffmpeg.input(input_path)
            
            logger.info(f"缩放视频: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            logger.info(f"  原始尺寸: {original_width}x{original_height}")
            logger.info(f"  目标尺寸: {target_width}x{new_height}")
            logger.info(f"  有音频: {video_info['has_audio']}")
            
            if video_info['has_audio']:
                # 有音频的情况
                logger.info("  使用音频输出模式")
                (
                    ffmpeg
                    .output(
                        input_stream.video.filter('scale', target_width, new_height),
                        input_stream.audio,  # 保留音频
                        output_path, 
                        vcodec='libx264', 
                        acodec='aac', 
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细信息
                )
            else:
                # 没有音频的情况
                logger.info("  使用无音频输出模式")
                (
                    ffmpeg
                    .output(
                        input_stream.video.filter('scale', target_width, new_height),
                        output_path, 
                        vcodec='libx264', 
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细信息
                )
            
            return True
        except Exception as e:
            logger.error(f"视频缩放失败 {input_path}: {str(e)}")
            return False
    
    def merge_videos_vertically(self, material_path, main_path, output_path, position="up", audio_mode="main_only", upper_audio_volume=1.0, lower_audio_volume=1.0, gif_path="", main_video_source="upper"):
        """垂直合并视频"""
        try:
            # 获取视频信息用于调试
            material_info = self.get_video_info(material_path)
            main_info = self.get_video_info(main_path)
            
            logger.info(f"垂直合并视频:")
            logger.info(f"  素材: {os.path.basename(material_path)} (有音频: {material_info['has_audio'] if material_info else '未知'})")
            logger.info(f"  主视频: {os.path.basename(main_path)} (有音频: {main_info['has_audio'] if main_info else '未知'})")
            logger.info(f"  位置: {position}, 音频模式: {audio_mode}")
            
            # 根据主视频来源确定位置关系
            # 上方视频文件夹的视频始终在上方，下方视频文件夹的视频始终在下方
            if main_video_source == "upper":
                # 主视频来自上方文件夹，素材视频来自下方文件夹
                # 上方视频在上，下方视频在下
                upper_input = ffmpeg.input(main_path)      # 上方视频（主视频）
                lower_input = ffmpeg.input(material_path)  # 下方视频（素材）
                logger.info(f"  位置关系: 上方视频({os.path.basename(main_path)}) 在上，下方视频({os.path.basename(material_path)}) 在下")
            else:
                # 素材视频来自上方文件夹，主视频来自下方文件夹
                # 上方视频在上，下方视频在下
                upper_input = ffmpeg.input(material_path)  # 上方视频（素材）
                lower_input = ffmpeg.input(main_path)      # 下方视频（主视频）
                logger.info(f"  位置关系: 上方视频({os.path.basename(material_path)}) 在上，下方视频({os.path.basename(main_path)}) 在下")
            
            # 使用vstack filter合并视频（上方视频在上，下方视频在下）
            video_output = ffmpeg.filter([upper_input.video, lower_input.video], 'vstack', inputs=2)
            
            # 检查是否需要叠加GIF
            if gif_path and gif_path.strip() and os.path.exists(gif_path.strip()):
                logger.info(f"  检测到GIF文件: {os.path.basename(gif_path)}")
                
                # 获取GIF信息
                gif_info = self.get_video_info(gif_path.strip())
                if not gif_info:
                    logger.warning(f"  警告: 无法获取GIF信息，跳过GIF叠加")
                else:
                    # 获取合并后视频的尺寸和时长
                    material_height = material_info['height']
                    main_height = main_info['height']
                    video_width = material_info['width']  # 两个视频宽度相同
                    total_height = material_height + main_height
                    
                    logger.info(f"  素材视频高度: {material_height}")
                    logger.info(f"  主视频高度: {main_height}")
                    logger.info(f"  合并后总高度: {total_height}")
                    logger.info(f"  视频宽度: {video_width}")
                    
                    # 计算合并后视频的总时长（使用主视频时长）
                    material_duration = material_info['duration']
                    main_duration = main_info['duration']
                    total_duration = main_duration  # 使用主视频时长作为目标时长
                    
                    logger.info(f"  主视频时长: {main_duration:.2f}秒")
                    logger.info(f"  素材视频时长: {material_duration:.2f}秒")
                    logger.info(f"  目标合并时长: {total_duration:.2f}秒")
                    logger.info(f"  GIF原始时长: {gif_info['duration']:.2f}秒")
                    
                    # 计算GIF缩放后的高度（等比缩放）
                    gif_original_width = gif_info['width']
                    gif_original_height = gif_info['height']
                    gif_new_height = int((video_width * gif_original_height) / gif_original_width)
                    
                    # 确保高度是偶数（视频编码要求）
                    if gif_new_height % 2 != 0:
                        gif_new_height += 1
                    
                    logger.info(f"  GIF原始尺寸: {gif_original_width}x{gif_original_height}")
                    logger.info(f"  GIF缩放尺寸: {video_width}x{gif_new_height}")
                    
                    # 创建GIF输入
                    gif_input = ffmpeg.input(gif_path.strip())
                    
                    # 使用loop filter实现真正的循环播放
                    gif_looped = gif_input.video.filter('loop', loop=-1, size=32767, start=0)
                    
                    # 缩放GIF到与主视频相同的宽度
                    gif_scaled = gif_looped.filter('scale', video_width, gif_new_height)
                    
                    # 计算GIF位置：中心与上下视频分割线重合
                    # 关键：分割线位置 = 上方视频的高度
                    # 根据实际的视频布局确定上方视频的高度
                    if main_video_source == "upper":
                        # 主视频来自上方文件夹，在上方；素材视频来自下方文件夹，在下方
                        upper_video_height = main_height
                        lower_video_height = material_height
                        logger.info(f"  布局: 主视频在上方({main_height}px)，素材视频在下方({material_height}px)")
                    else:
                        # 素材视频来自上方文件夹，在上方；主视频来自下方文件夹，在下方
                        upper_video_height = material_height
                        lower_video_height = main_height
                        logger.info(f"  布局: 素材视频在上方({material_height}px)，主视频在下方({main_height}px)")
                    
                    # 分割线位置就是上方视频的高度
                    seam_position = upper_video_height
                    
                    # 使用比例计算确保精确对齐：y = H * (seam_position / total_height) - h/2
                    seam_ratio = seam_position / total_height
                    gif_y_expr = f'H*{seam_ratio:.10f} - h / 2'
                    
                    logger.info(f"  分割线位置: {seam_position}px (比例: {seam_ratio:.6f})")
                    logger.info(f"  GIF缩放尺寸: {video_width}x{gif_new_height}")
                    logger.info(f"  GIF位置表达式: {gif_y_expr}")

                    video_output = ffmpeg.filter(
                        [video_output, gif_scaled],
                        'overlay',
                        x='(W-w)/2',   # 水平居中
                        y=gif_y_expr,   # GIF 中心与分割线重合（使用比例计算）
                        shortest=1,
                    )
                    logger.info("  GIF循环播放设置: 使用loop filter实现无限循环，输出时长由主视频决定")
                    logger.info(f"  GIF叠加位置: 水平居中，中心与分割线重合 (y={gif_y_expr})")
            
            # 根据音频模式处理音频
            if audio_mode == "mix":
                # 混音模式：先检查音频状态
                logger.info(f"  混音模式 - 上方视频音量: {upper_audio_volume}, 下方视频音量: {lower_audio_volume}")
                
                # 检查素材和主视频的音频状态
                material_info = self.get_video_info(material_path)
                main_info = self.get_video_info(main_path)
                
                if not material_info or not main_info:
                    logger.warning("  无法获取视频信息，使用下方视频音频(应用音量)")
                    audio_output = lower_input.audio.filter('volume', lower_audio_volume)
                elif not material_info['has_audio'] and not main_info['has_audio']:
                    # 两个视频都没有音频
                    logger.error("  错误：素材视频和主视频都没有音频，无法进行混音处理")
                    raise ValueError("素材视频和主视频都没有音频，无法进行混音处理")
                elif not material_info['has_audio']:
                    # 只有主视频有音频
                    logger.info("  素材视频没有音频，使用下方视频音频(应用音量)")
                    audio_output = lower_input.audio.filter('volume', lower_audio_volume)
                elif not main_info['has_audio']:
                    # 只有素材视频有音频
                    logger.info("  主视频没有音频，使用上方视频音频(应用音量)")
                    audio_output = upper_input.audio.filter('volume', upper_audio_volume)
                else:
                    # 两个视频都有音频，进行混音
                    logger.info("  两个视频都有音频，进行混音处理")
                    # 对上方视频音频应用音量调整
                    upper_audio_adjusted = upper_input.audio.filter('volume', upper_audio_volume)
                    # 对下方视频音频应用音量调整
                    lower_audio_adjusted = lower_input.audio.filter('volume', lower_audio_volume)
                    
                    # 使用amix filter混合音频
                    audio_output = ffmpeg.filter([upper_audio_adjusted, lower_audio_adjusted], 'amix', inputs=2, duration='longest')
                    logger.info("  混音模式：成功创建混音")
            else:
                # 只使用主视频音频
                logger.info("  使用主视频音频")
                audio_output = lower_input.audio
            
            # 输出合并后的视频
            (
                ffmpeg
                .output(
                    video_output,
                    audio_output,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    audio_bitrate='128k',
                    preset='medium'
                )
                .overwrite_output()
                .run(quiet=False)  # 显示详细错误信息
            )
            
            return True
            
        except Exception as e:
            logger.error(f"视频合并失败: {str(e)}")
            return False
    
    def merge_videos(self, upper_video_folder: str, lower_video_folder: str, main_video_source: str, 
                     upper_video_volume: float, lower_video_volume: float, output_folder_prefix: str, gif_path: str = ""):
        """
        合并视频文件
        
        Args:
            upper_video_folder: 上方视频文件夹路径（支持相对路径和绝对路径）
            lower_video_folder: 下方视频文件夹路径（支持相对路径和绝对路径）
            main_video_source: 主体视频选择（upper/lower）
            upper_video_volume: 上方视频音频音量
            lower_video_volume: 下方视频音频音量
            output_folder_prefix: 输出文件夹前缀
            gif_path: GIF文件夹路径（可选，支持相对路径和绝对路径，GIF文件与主视频依次绑定，不足时循环使用）
        """
        try:
            # 默认使用mix模式
            audio_mode = "mix"
            
            # 音量直接对应上方视频和下方视频
            upper_audio_volume = upper_video_volume
            lower_audio_volume = lower_video_volume
            
            logger.info(f"音量设置 - 上方视频音量: {upper_audio_volume}, 下方视频音量: {lower_audio_volume}")
            
            # 解析输入路径（支持相对路径和绝对路径）
            upper_input_path = resolve_path(upper_video_folder)
            lower_input_path = resolve_path(lower_video_folder)
            
            # 验证输入路径是否存在
            if not os.path.exists(upper_input_path):
                return ("",)
            if not os.path.exists(lower_input_path):
                return ("",)
            
            # 根据主体视频选择确定主视频和素材视频路径
            # 位置关系固定：上方视频文件夹始终在上方，下方视频文件夹始终在下方
            if main_video_source == "upper":
                # 上方视频作为主视频，下方视频作为素材视频
                main_video_path = upper_input_path
                material_video_path = lower_input_path
            else:
                # 下方视频作为主视频，上方视频作为素材视频
                main_video_path = lower_input_path
                material_video_path = upper_input_path
            
            # 位置关系固定：上方视频在上，下方视频在下
            position = "up"  # 上方视频文件夹在上方，下方视频文件夹在下方
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            unique_folder_name = generate_unique_folder_name(output_folder_prefix, output_dir)
            output_path = os.path.join(output_dir, unique_folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 支持的视频格式
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            
            # 获取所有主视频文件
            main_videos = []
            for ext in video_extensions:
                pattern = os.path.join(main_video_path, ext)
                main_videos.extend(glob.glob(pattern))
            
            # 获取所有素材视频文件
            material_videos = []
            for ext in video_extensions:
                pattern = os.path.join(material_video_path, ext)
                material_videos.extend(glob.glob(pattern))
            
            # 获取所有GIF文件
            gif_files = []
            if gif_path and gif_path.strip():
                gif_folder_path = resolve_path(gif_path.strip())
                if os.path.exists(gif_folder_path):
                    gif_extensions = ['*.gif', '*.GIF']
                    for ext in gif_extensions:
                        pattern = os.path.join(gif_folder_path, ext)
                        gif_files.extend(glob.glob(pattern))
                    logger.info(f"找到 {len(gif_files)} 个GIF文件")
                else:
                    logger.warning(f"GIF文件夹不存在: {gif_folder_path}")
            
            if not main_videos:
                return ("",)
            if not material_videos:
                return ("",)
            
            # 预分配任务（串行），确保素材与 GIF 与原流程一致的顺序消费
            tasks = []
            material_index = 0
            gif_index = 0
            output_paths = []

            for main_video in main_videos:
                if material_index >= len(material_videos):
                    break  # 素材用完则结束

                main_info = self.get_video_info(main_video)
                if not main_info:
                    continue

                main_filename = Path(main_video).stem
                if audio_mode == "mix" and not main_info['has_audio']:
                    logger.warning(f"警告: 主视频 {main_filename} 没有音频，在mix模式下可能影响混音效果")

                current_gif_path = ""
                if gif_files:
                    current_gif_path = gif_files[gif_index % len(gif_files)]
                    gif_index += 1

                main_duration = main_info['duration']
                used_materials = []
                current_duration = 0

                while current_duration < main_duration and material_index < len(material_videos):
                    material_video = material_videos[material_index]
                    material_info = self.get_video_info(material_video)
                    material_index += 1
                    if not material_info:
                        continue
                    material_filename = Path(material_video).stem
                    if audio_mode == "mix" and not material_info['has_audio']:
                        logger.warning(f"警告: 素材视频 {material_filename} 没有音频，在mix模式下可能影响混音效果")
                    used_materials.append({'path': material_video, 'duration': material_info['duration'], 'info': material_info})
                    current_duration += material_info['duration']

                if not used_materials:
                    continue

                output_file = os.path.join(output_path, f"{main_filename}_merged.mp4")
                tasks.append({
                    'main_video': main_video,
                    'main_info': main_info,
                    'main_duration': main_duration,
                    'output_file': output_file,
                    'gif_path': current_gif_path,
                    'used_materials': used_materials,
                    'main_video_source': main_video_source,
                    'position': position,
                    'audio_mode': audio_mode,
                    'upper_audio_volume': upper_audio_volume,
                    'lower_audio_volume': lower_audio_volume,
                })

            if not tasks:
                return ("",)

            # 并发处理每个主视频任务
            def process_task(task):
                import threading
                thread_id = threading.current_thread().ident
                
                main_video = task['main_video']
                main_info = task['main_info']
                main_duration = task['main_duration']
                output_file = task['output_file']
                current_gif_path = task['gif_path']
                used_materials = task['used_materials']
                main_video_source = task['main_video_source']
                position = task['position']
                audio_mode = task['audio_mode']
                upper_audio_volume = task['upper_audio_volume']
                lower_audio_volume = task['lower_audio_volume']

                try:
                    main_filename = Path(main_video).stem
                    if current_gif_path:
                        logger.info(f"[线程{thread_id}] 主视频 {main_filename} 使用GIF: {os.path.basename(current_gif_path)}")

                    # 使用线程ID确保临时目录唯一性
                    temp_dir = tempfile.mkdtemp(prefix=f"merge_{thread_id}_")
                    temp_material_path = os.path.join(temp_dir, f"temp_material_{main_filename}_{thread_id}.mp4")

                    main_width = main_info['width']

                    if audio_mode == "mix":
                        materials_with_audio = [m for m in used_materials if m['info']['has_audio']]
                        materials_without_audio = [m for m in used_materials if not m['info']['has_audio']]
                        if not main_info['has_audio'] and not materials_with_audio:
                            logger.error(f"错误: 主视频 {main_filename} 和所有素材视频都没有音频，无法进行混音处理")
                            return None
                        elif not main_info['has_audio']:
                            logger.warning(f"警告: 主视频 {main_filename} 没有音频，将只使用素材音频")
                        elif not materials_with_audio:
                            logger.warning(f"警告: 所有素材视频都没有音频，将只使用主视频音频")
                        elif materials_without_audio:
                            logger.warning(f"警告: {len(materials_without_audio)} 个素材视频没有音频，可能影响混音效果")

                    resized_materials = []
                    for i, material in enumerate(used_materials):
                        resized_path = os.path.join(temp_dir, f"resized_material_{i}_{thread_id}.mp4")
                        if self.resize_video_to_width(material['path'], main_width, resized_path):
                            if material['info']['has_audio']:
                                resized_materials.append(resized_path)
                            else:
                                try:
                                    resized_with_audio = os.path.join(temp_dir, f"resized_material_{i}_a_{thread_id}.mp4")
                                    silence = (ffmpeg.input('anullsrc=channel_layout=stereo:sample_rate=44100', f='lavfi', t=material['duration']).audio)
                                    video_stream = ffmpeg.input(resized_path).video
                                    (ffmpeg.output(video_stream, silence, resized_with_audio, vcodec='libx264', acodec='aac', preset='medium').overwrite_output().run(quiet=True))
                                    resized_materials.append(resized_with_audio)
                                    try:
                                        os.remove(resized_path)
                                    except:
                                        pass
                                except Exception:
                                    resized_materials.append(resized_path)

                    if not resized_materials:
                        return None

                    concat_file = os.path.join(temp_dir, f"concat_list_{main_filename}_{thread_id}.txt")
                    with open(concat_file, 'w') as f:
                        for resized_material in resized_materials:
                            f.write(f"file '{resized_material}'\n")

                    has_any_audio = any(m['info']['has_audio'] for m in used_materials)
                    if has_any_audio:
                        (ffmpeg.input(concat_file, format='concat', safe=0).output(temp_material_path, vcodec='libx264', acodec='aac', preset='medium').overwrite_output().run(quiet=True))
                    else:
                        (ffmpeg.input(concat_file, format='concat', safe=0).output(temp_material_path, vcodec='libx264', preset='medium').overwrite_output().run(quiet=True))

                    try:
                        os.remove(concat_file)
                        for resized_material in resized_materials:
                            os.remove(resized_material)
                    except:
                        pass

                    temp_material_info = self.get_video_info(temp_material_path)
                    if not temp_material_info:
                        logger.warning(f"  警告: 无法获取合并后素材视频信息，跳过主视频: {main_filename}")
                        return None

                    temp_material_duration = temp_material_info['duration']
                    if temp_material_duration < main_duration:
                        logger.warning(f"  警告: 合并后素材时长 ({temp_material_duration:.2f}秒) 不足以支持主视频时长 ({main_duration:.2f}秒)，跳过主视频: {main_filename}")
                        return None

                    logger.info(f"  合并后素材时长: {temp_material_duration:.2f}秒，主视频时长: {main_duration:.2f}秒")

                    temp_material_cropped = os.path.join(temp_dir, f"temp_material_cropped_{main_filename}_{thread_id}.mp4")
                    if temp_material_info and temp_material_info['has_audio']:
                        (ffmpeg.input(temp_material_path, t=main_duration).output(temp_material_cropped, vcodec='libx264', acodec='aac', preset='medium').overwrite_output().run(quiet=True))
                    else:
                        (ffmpeg.input(temp_material_path, t=main_duration).output(temp_material_cropped, vcodec='libx264', preset='medium').overwrite_output().run(quiet=True))
                    os.remove(temp_material_path)
                    temp_material_path = temp_material_cropped

                    if self.merge_videos_vertically(temp_material_path, main_video, output_file, position, audio_mode, upper_audio_volume, lower_audio_volume, current_gif_path, main_video_source):
                        logger.info(f"[线程{thread_id}] 成功合并: {main_filename} -> {output_file}")
                        return output_file
                    return None
                except Exception as e:
                    logger.error(f"[线程{thread_id}] 处理主视频 {main_video} 时出错: {str(e)}")
                    return None
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass

            processed_count = 0
            # 降低并发数避免资源竞争，特别是ffmpeg进程冲突
            max_workers = min(4, max(1, os.cpu_count() - 2 or 2))
            logger.info(f"使用 {max_workers} 个线程并发处理 {len(tasks)} 个任务")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(process_task, t): t for t in tasks}
                for future in as_completed(future_to_task):
                    result_path = future.result()
                    if result_path:
                        processed_count += 1
                        output_paths.append(result_path)
 
            if processed_count == 0:
                return ("",)  # 没有可保存的视频时返回空字符串
            else:
                # 返回输出文件的目录路径
                logger.info(f"成功处理 {processed_count} 个主视频")
                logger.info(f"输出目录: {output_path}")
                logger.info(f"输出文件: {output_paths}")
                return (output_path,)
                
        except ValueError as e:
            # 输入验证错误
            return ("",)
        except Exception as e:
            # 其他错误
            return ("",)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoMergeNode": VideoMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMergeNode": "视频画幅合并"
}