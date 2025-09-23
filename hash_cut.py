import cv2
import imagehash
from PIL import Image
import numpy as np
import json
import os
import ffmpeg
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import folder_paths
from pathlib import Path
import tempfile
import shutil

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm", "*.m4v"]

def get_video_files(directory):
    """获取目录中所有支持的视频文件"""
    video_files = []
    for fmt in SUPPORTED_VIDEO_FORMATS:
        pattern = os.path.join(directory, fmt)
        video_files.extend(glob.glob(pattern))
    return video_files

# --- Step1: 计算单帧的pHash ---
def phash_frame(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)

def normalize_video_resolution(recording_file, clip_dir, target_resolution=None):
    """
    统一录播视频和片段文件夹中所有视频的分辨率
    
    参数:
        recording_file: 录播视频文件路径
        clip_dir: 片段文件夹路径
        target_resolution: 目标分辨率，如果为None则自动使用录播视频的分辨率
    
    返回:
        tuple: (统一后的录播视频路径, 统一后的片段文件夹路径, 临时文件夹路径)
    """
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp(prefix="video_normalize_")
    temp_clip_dir = os.path.join(temp_dir, "clips")
    os.makedirs(temp_clip_dir, exist_ok=True)
    
    print(f"创建临时文件夹: {temp_dir}")
    
    try:
        # 0. 自动检测录播视频分辨率（如果未指定目标分辨率）
        if target_resolution is None:
            recording_resolution = get_video_resolution(recording_file)
            if recording_resolution:
                target_resolution = recording_resolution
                print(f"自动检测到录播视频分辨率: {target_resolution[0]}x{target_resolution[1]}")
            else:
                target_resolution = (1920, 1080)
                print(f"无法检测录播视频分辨率，使用默认分辨率: {target_resolution[0]}x{target_resolution[1]}")
        else:
            print(f"使用指定分辨率: {target_resolution[0]}x{target_resolution[1]}")
        
        # 1. 处理录播视频
        print(f"正在统一录播视频分辨率: {recording_file}")
        recording_basename = os.path.basename(recording_file)
        temp_recording_file = os.path.join(temp_dir, recording_basename)
        
        # 检查录播视频是否需要转换分辨率
        recording_resolution = get_video_resolution(recording_file)
        if recording_resolution and recording_resolution == target_resolution:
            print("录播视频分辨率已匹配，创建软连接")
            os.symlink(recording_file, temp_recording_file)
        else:
            # 使用ffmpeg统一录播视频分辨率
            (
                ffmpeg
                .input(recording_file)
                .filter('scale', target_resolution[0], target_resolution[1])
                .output(temp_recording_file, vcodec='libx264', acodec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
        print(f"录播视频分辨率已统一: {temp_recording_file}")
        
        # 2. 处理片段文件夹中的所有视频
        print(f"正在统一片段文件夹中视频的分辨率: {clip_dir}")
        processed_clips = []
        
        for clip_file in os.listdir(clip_dir):
            if clip_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                clip_file_path = os.path.join(clip_dir, clip_file)
                temp_clip_file = os.path.join(temp_clip_dir, clip_file)
                
                try:
                    # 检查片段视频是否需要转换分辨率
                    clip_resolution = get_video_resolution(clip_file_path)
                    if clip_resolution and clip_resolution == target_resolution:
                        print(f"片段视频 {clip_file} 分辨率已匹配，创建软连接")
                        os.symlink(clip_file_path, temp_clip_file)
                    else:
                        # 使用ffmpeg统一片段视频分辨率
                        (
                            ffmpeg
                            .input(clip_file_path)
                            .filter('scale', target_resolution[0], target_resolution[1])
                            .output(temp_clip_file, vcodec='libx264', acodec='copy')
                            .overwrite_output()
                            .run(quiet=True)
                        )
                        print(f"片段视频分辨率已统一: {clip_file}")
                    
                    processed_clips.append(temp_clip_file)
                except Exception as e:
                    print(f"处理片段视频 {clip_file} 时出错: {str(e)}")
                    # 如果处理失败，创建软连接
                    try:
                        os.symlink(clip_file_path, temp_clip_file)
                    except:
                        # 如果软连接创建失败，则复制原文件
                        shutil.copy2(clip_file_path, temp_clip_file)
                    processed_clips.append(temp_clip_file)
        
        print(f"分辨率统一完成，共处理 {len(processed_clips)} 个片段视频")
        return temp_recording_file, temp_clip_dir, temp_dir
        
    except Exception as e:
        print(f"统一视频分辨率时发生错误: {str(e)}")
        # 清理临时文件夹
        cleanup_temp_folder(temp_dir)
        raise e

def cleanup_temp_folder(temp_dir):
    """
    清理临时文件夹
    
    参数:
        temp_dir: 临时文件夹路径
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"临时文件夹已清理: {temp_dir}")
    except Exception as e:
        print(f"清理临时文件夹时出错: {str(e)}")

def get_video_resolution(video_file):
    """
    获取视频文件的分辨率
    
    参数:
        video_file: 视频文件路径
    
    返回:
        tuple: (宽度, 高度)
    """
    try:
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            return (width, height)
        else:
            return None
    except Exception as e:
        print(f"获取视频分辨率时出错: {str(e)}")
        return None

# 使用示例：
"""
# 示例1：自动检测录播视频分辨率并统一所有视频
temp_recording_file, temp_clip_dir, temp_dir = normalize_video_resolution(
    recording_file="path/to/recording.mp4",
    clip_dir="path/to/clips/",
    target_resolution=None  # 自动使用录播视频的分辨率
)

# 示例2：手动指定目标分辨率
temp_recording_file, temp_clip_dir, temp_dir = normalize_video_resolution(
    recording_file="path/to/recording.mp4",
    clip_dir="path/to/clips/",
    target_resolution=(1920, 1080)  # 手动指定分辨率
)

# 使用完毕后记得清理临时文件夹
cleanup_temp_folder(temp_dir)
"""


def phash_frame_hd(frame):
    """高精度感知哈希（更高位宽，更敏感，非兼容阈值）
    - 灰度化 + CLAHE 提升对比度稳定性
    - 轻度锐化（Unsharp Mask）增强微小边缘/纹理
    - 使用更大 hash_size 与更高 highfreq_factor 提升区分度（位宽↑）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Unsharp mask: 强化细节，增大对微小变化的响应
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    sharp = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)
    pil_img = Image.fromarray(sharp)
    # 使用 16x16（256 bit）并提高高频采样，提升微变化敏感度
    return imagehash.phash(pil_img, hash_size=16, highfreq_factor=8)


def hash_distance_norm(h1, h2):
    """归一化汉明距离: 返回 [0,1]，与位宽无关。"""
    bits = float(h1.hash.size)
    return (h1 - h2) / bits

# --- Step2: 从视频中提取帧的哈希序列 ---
def video_to_phash(video_path):
    """按秒采样，默认每 1 秒取一帧并计算 pHash。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    # 获取视频总时长
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    interval_sec = 1.0
    hashes, timestamps = [], []

    # 直接跳转到指定时间点进行采样
    sample_time = 0.0
    while sample_time < duration:
        # 设置到指定时间点
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        h = phash_frame(frame)
        hashes.append(h)
        timestamps.append(sample_time)
        sample_time += interval_sec
    
    cap.release()
    return hashes, timestamps


def detect_static_segments(video_path, diff_threshold=1.0, min_static_seconds=2.0):
    """检测连续静止的时间段（单位：秒）。
    - diff_threshold: 帧间 pHash 距离阈值，小于等于该阈值视为无显著变化
    - min_static_seconds: 静止持续时长下限（秒），短于该值不计
    返回：[(start_sec, end_sec), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # 获取视频总时长
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    interval_sec = 1.0  # 每1秒检测一次
    
    prev_hash = None
    static_start = None
    static_ranges = []

    # 直接跳转到指定时间点进行采样
    sample_time = 0.0
    while sample_time < duration:
        # 设置到指定时间点
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        cur_hash = phash_frame_hd(frame)
        if prev_hash is not None:
            dist = hash_distance_norm(prev_hash, cur_hash)
            if dist <= diff_threshold:
                if static_start is None:
                    static_start = sample_time
            else:
                if static_start is not None:
                    duration_static = sample_time - static_start
                    if duration_static >= min_static_seconds:
                        static_ranges.append((static_start, sample_time))
                static_start = None
        prev_hash = cur_hash
        sample_time += interval_sec

    # 收尾
    if static_start is not None:
        if duration - static_start >= min_static_seconds:
            static_ranges.append((static_start, duration))
    
    cap.release()
    return static_ranges

def _probe_duration_seconds(video_path):
    try:
        info = ffmpeg.probe(video_path)
        duration = float(info['format']['duration'])
        return duration
    except Exception:
        return None


def _invert_segments(segments, duration):
    """将需要剔除的区间反转为需要保留的区间。"""
    if duration is None:
        return []
    if not segments:
        return [(0.0, duration)]
    segments = sorted(segments, key=lambda x: x[0])
    keep = []
    prev = 0.0
    for s, e in segments:
        s = max(0.0, s)
        e = min(duration, e)
        if s > prev:
            keep.append((prev, s))
        prev = max(prev, e)
    if prev < duration:
        keep.append((prev, duration))
    return [(round(s, 3), round(e, 3)) for s, e in keep if e - s > 1e-3]

def export_video_without_segments(video_path, segments, output_path):
    """使用 ffmpeg 将给定静止区间删除并导出新视频。
    segments: [(start_sec, end_sec), ...] 按时间升序且不重叠
    """
    file_name = os.path.basename(video_path)
    file_name_list = os.path.splitext(file_name)
    file_name = file_name_list[0] + "_without_segments." + file_name_list[1]
    output_path = os.path.join(output_path, file_name)
    duration = _probe_duration_seconds(video_path)
    keep_ranges = _invert_segments(segments, duration)
    if not keep_ranges:
        # 无法计算时长或异常，直接复制封装（尽量保持成功）
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, c='copy', movflags='faststart')
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        except Exception:
            return False

    # 构造 select/aselect 表达式，保留区间并重置时间戳
    expr = '+'.join([f'between(t,{s},{e})' for s, e in keep_ranges])

    try:
        inp = ffmpeg.input(video_path)

        # 检查是否有音频流
        has_audio = False
        try:
            info = ffmpeg.probe(video_path)
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    has_audio = True
                    break
        except Exception:
            has_audio = False

        v = inp.video.filter('select', expr).filter('setpts', 'N/FRAME_RATE/TB')
        if has_audio:
            a = inp.audio.filter('aselect', expr).filter('asetpts', 'N/SR/TB')
            out = ffmpeg.output(
                v, a, output_path,
                vcodec='libx264', acodec='aac', movflags='faststart',
                video_bitrate='3000k', audio_bitrate='192k', vsync='vfr'
            )
        else:
            out = ffmpeg.output(
                v, output_path,
                vcodec='libx264', movflags='faststart',
                video_bitrate='3000k', vsync='vfr'
            )

        out = out.overwrite_output()
        out.run(quiet=True)
        return True
    except Exception:
        return False

# --- Step3: 匹配视频片段 ---
def match_video(query_hashes, target_hashes, threshold=5):
    """找到所有匹配的视频片段"""
    qlen = len(query_hashes)
    matches = []
    
    for i in range(len(target_hashes) - qlen + 1):
        # 计算窗口内的平均汉明距离（改进哈希后直接沿用）
        dist = np.mean([query_hashes[j] - target_hashes[i+j] for j in range(qlen)])
        if dist <= threshold:
            matches.append((i, dist))
    
    # 按距离排序，距离越小越相似
    matches.sort(key=lambda x: x[1])
    return matches

def match_video_single(query_hashes, target_hashes):
    """找到最佳匹配位置（原函数）"""
    qlen = len(query_hashes)
    best_score, best_pos = float('inf'), -1

    for i in range(len(target_hashes) - qlen + 1):
        # 计算窗口内的平均汉明距离
        dist = np.mean([query_hashes[j] - target_hashes[i+j] for j in range(qlen)])
        if dist < best_score:
            best_score, best_pos = dist, i

    return best_pos, best_score

def filter_overlapping_matches(matches, min_gap=5):
    """合并相邻/重叠的匹配窗口。
    - 输入 matches: [(pos, score), ...]，pos 为起始索引（按秒采样即秒）
    - 将起点彼此间隔 <= min_gap 的窗口归并为一个区间
    - 返回: [(start_pos, end_pos, score), ...]，score 取区间内最优（较小）
    注意：end_pos 是“窗口起点索引”，最终真实结束帧索引应为 end_pos + query_length - 1。
    """
    if not matches:
        return []

    # 先按起点排序，忽略原来的按得分排序
    matches_sorted = sorted(matches, key=lambda x: x[0])

    merged = []
    cur_start, cur_end, cur_score = None, None, None

    for pos, score in matches_sorted:
        if cur_start is None:
            cur_start, cur_end, cur_score = pos, pos, score
            continue

        # 如果新窗口的起点与当前区间的最后一个起点足够近，则合并
        if pos <= cur_end + min_gap:
            cur_end = max(cur_end, pos)
            cur_score = min(cur_score, score)
        else:
            merged.append((cur_start, cur_end, cur_score))
            cur_start, cur_end, cur_score = pos, pos, score

    if cur_start is not None:
        merged.append((cur_start, cur_end, cur_score))

    return merged

def build_segments(matches, ts_target, query_len):
    """
    将匹配结果转换为时间片段
    
    功能：将视频匹配的索引位置转换为实际的时间片段
    参数：
        - matches: 匹配结果列表，每个元素包含位置和相似度分数
        - ts_target: 目标视频的时间戳数组
        - query_len: 查询视频的长度（帧数）
    返回：
        - segments: 时间片段列表，每个片段包含开始时间、结束时间和相似度分数
    """
    segments = []
    
    # 遍历所有匹配结果
    for item in matches:
        # 处理两种不同的匹配结果格式
        if len(item) == 2:
            # 格式1：(位置, 分数) - 单点匹配
            pos, score = item
            start_index = pos
            end_index = min(pos + query_len - 1, len(ts_target) - 1)
        else:
            # 格式2：(开始位置, 结束位置, 分数) - 范围匹配
            start_pos, end_pos, score = item
            start_index = start_pos
            end_index = min(end_pos + query_len - 1, len(ts_target) - 1)

        # 确保索引在有效范围内，防止越界
        start_index = max(0, min(start_index, len(ts_target) - 1))
        end_index = max(0, min(end_index, len(ts_target) - 1))

        # 将索引转换为实际时间戳
        start_time = ts_target[start_index]
        end_time = ts_target[end_index]
        
        # 构建时间片段对象
        segments.append({
            "start": float(start_time),    # 开始时间（秒）
            "end": float(end_time),        # 结束时间（秒）
            "score": float(score)          # 相似度分数
        })
    
    return segments

def merge_segments(all_segments, gap):
    """
    合并相邻的时间片段，减少碎片化
    
    功能：将间隔小于指定阈值的相邻时间片段合并为一个片段
    参数：
        - all_segments: 时间片段列表，每个片段包含start、end、score字段
        - gap: 合并阈值（秒），小于此间隔的片段会被合并
    返回：
        - merged: 合并后的时间片段列表
    """
    # 输入验证
    if not all_segments:
        return []
    
    # 验证片段格式
    for seg in all_segments:
        if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
            print(f"警告：跳过无效的时间片段格式: {seg}")
            continue
        if seg["start"] > seg["end"]:
            print(f"警告：跳过时间范围无效的片段: {seg}")
            continue
    
    # 按开始时间排序
    all_segments = sorted(all_segments, key=lambda x: (x["start"], x["end"]))
    
    # 开始合并
    merged = [all_segments[0].copy()]  # 复制第一个片段避免修改原数据
    
    for seg in all_segments[1:]:
        last = merged[-1]
        
        # 检查是否需要合并：当前片段开始时间 <= 上一个片段结束时间 + 合并间隔
        if seg["start"] <= last["end"] + gap + 1e-6:  # 1e-6用于处理浮点数精度问题
            # 合并片段：扩展结束时间到较晚的那个
            last["end"] = max(last["end"], seg["end"])
            # 保留更高的相似度分数（相似度越高越好）
            last["score"] = max(last.get("score", 0.0), seg.get("score", 0.0))
        else:
            # 间隔太大，添加为新片段
            merged.append({
                "start": seg["start"], 
                "end": seg["end"], 
                "score": seg.get("score", 0.0)
            })
    
    return merged


class ComfyNodeABC:
    """ComfyUI 节点基类"""
    pass


class VideoHashCutNode(ComfyNodeABC):
    """基于感知哈希的视频静止片段检测与移除节点"""
    
    @classmethod
    def get_input_folders(cls):
        """获取输入目录下的所有子文件夹"""
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return ["input"]
            
            folders = ["input"]  # 默认包含根目录
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)
            
            return sorted(folders)
        except Exception:
            return ["input"]
    
    @classmethod
    def get_target_folders(cls):
        """获取目标文件夹列表（输入目录下的所有子文件夹）"""
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return []
            
            folders = []
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)
            
            return sorted(folders)
        except Exception:
            return []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": (cls.get_input_folders(), {
                    "default": "input",
                    "tooltip": "选择输入视频文件夹"
                }),
                "video_clip_path": (cls.get_target_folders(), {
                    "default": "",
                    "tooltip": "选择需要删除的视频片段文件夹"
                }),
                "output_folder_name": ("STRING", {
                    "default": "hash_cut_output",
                    "tooltip": "输出文件夹名称"
                }),
                "match_threshold": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "视频匹配阈值"
                }),
                "max_workers": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "最大线程数"
                })
            },
            "optional": {
                "custom_input_path": ("STRING", {
                    "default": "",
                    "tooltip": "自定义输入路径（优先级高于下拉选择）"
                }),
                "custom_clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "自定义视频片段路径（优先级高于下拉选择）"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    CATEGORY = "video_editing"
    
    def execute(self, input_folder, output_folder_name, match_threshold, max_workers, video_clip_path, custom_input_path="", custom_clip_path=""):
        """
        执行视频静止片段检测与移除
        """
        # 设置默认参数值（这些参数不在UI中显示，使用优化后的默认值）
        min_gap_seconds = 5.0          # 匹配结果间最小间隔（秒），避免过于频繁的片段分割
        merge_gap_seconds = 1.0         # 合并片段的最大间隔（秒），小于此间隔的片段会被合并
        # 注意：已移除静止片段检测相关参数，简化处理流程
        
        try:
            
            # 解析录播视频路径
            if custom_input_path and os.path.exists(custom_input_path):
                recording_video_path = custom_input_path
            else:
                recording_video_path = folder_paths.get_input_directory()
                recording_video_path = os.path.join(recording_video_path, input_folder)
            
            if not os.path.exists(recording_video_path):
                print(f"录播视频文件夹不存在: {recording_video_path}")
                return ("",)
            
            # 创建输出目录
            output_dir = folder_paths.get_output_directory()
            output_dir = os.path.join(output_dir, output_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 解析视频片段路径（支持自定义路径）
            if custom_clip_path and os.path.exists(custom_clip_path):
                clip_dir = custom_clip_path
                print(f"使用自定义视频片段路径: {clip_dir}")
            else:
                # 检查视频片段路径是否指定
                if not video_clip_path:
                    print("请选择需要删除的视频片段文件夹")
                    return ("",)
                
                # 构建视频片段文件夹路径
                input_dir = folder_paths.get_input_directory()
                clip_dir = os.path.join(input_dir, video_clip_path)
                if not os.path.exists(clip_dir):
                    print(f"视频片段文件夹不存在: {clip_dir}")
                    return ("",)
                
                print(f"处理视频片段文件夹: {video_clip_path}")
            
            # 获取视频片段文件（支持多种格式）
            clip_files = get_video_files(clip_dir)
            
            if not clip_files:
                print(f"视频片段文件夹 {video_clip_path} 中没有找到视频文件")
                return ("",)
            
            # 准备任务列表 - 为录播视频文件夹中的每个文件创建处理任务
            tasks = []
            for re_file in os.listdir(recording_video_path):
                # 构建任务参数：每个录播视频文件与片段文件夹进行匹配
                task_params = {
                    'recording_file': os.path.join(recording_video_path, re_file),  # 单个录播视频文件路径
                    'clip_dir': clip_dir,                                          # 视频片段文件夹路径
                    'match_threshold': match_threshold,                            # 视频匹配阈值
                    'min_gap_seconds': min_gap_seconds,                           # 匹配结果间最小间隔
                    'merge_gap_seconds': merge_gap_seconds,                        # 合并片段的最大间隔
                    'output_dir': output_dir                                      # 输出文件夹路径
                }
                tasks.append(task_params)
                print(f"  添加任务: {video_clip_path} -> {os.path.basename(re_file)}")
            
            if not tasks:
                print("没有找到有效的处理任务")
                return ("",)
            
            print(f"准备处理 {len(tasks)} 个任务")
            
            # 创建线程锁用于输出同步
            lock = threading.Lock()
            
            # 使用线程池执行任务
            results = []
            print(f"启动线程池，最大工作线程数: {max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(self.process_single_video_with_params, task): task 
                    for task in tasks
                }
                
                print(f"已提交 {len(future_to_task)} 个任务到线程池")
                
                # 收集结果
                completed_count = 0
                print("开始等待任务完成...")
                for future in as_completed(future_to_task):
                    completed_count += 1
                    task = future_to_task[future]
                    game_name = os.path.basename(task['recording_file'])
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            with lock:
                                print(f"[进度 {completed_count}/{len(tasks)}] {game_name} 处理完成")
                        else:
                            with lock:
                                print(f"[进度 {completed_count}/{len(tasks)}] {game_name} 处理失败")
                    except Exception as e:
                        with lock:
                            print(f"[进度 {completed_count}/{len(tasks)}] {game_name} 异常: {str(e)}")

            return (output_dir,)
            
        except Exception as e:
            print(f"处理视频时发生错误: {str(e)}")
            return ("",)
    
    def process_single_video_with_params(self, task_params):
        """
        处理单个录播视频的重复片段检测任务
        
        功能：将录播视频与片段文件夹中的所有视频进行匹配，找出重复片段
        参数：
            - recording_file: 单个录播视频文件路径
            - clip_dir: 视频片段文件夹路径（包含需要删除的片段）
            - match_threshold: 匹配阈值，用于判断相似度
            - min_gap_seconds: 匹配结果间最小间隔
            - merge_gap_seconds: 合并片段的最大间隔
        """
        # 从字典中提取参数
        recording_file = task_params['recording_file']      # 录播视频文件路径
        clip_dir = task_params['clip_dir']                  # 片段视频文件夹路径
        match_threshold = task_params['match_threshold']    # 视频匹配阈值
        min_gap_seconds = task_params['min_gap_seconds']    # 匹配结果间最小间隔
        merge_gap_seconds = task_params['merge_gap_seconds'] # 合并片段的最大间隔
        output_dir = task_params['output_dir']              # 输出文件夹路径
        
        temp_dir = None
        try:
            # 步骤0：统一视频分辨率
            print("开始统一视频分辨率...")
            temp_recording_file, temp_clip_dir, temp_dir = normalize_video_resolution(
                recording_file, clip_dir, target_resolution=None  # 自动检测录播视频分辨率
            )
            
            # 步骤1：计算录播视频的感知哈希和时间戳
            target_hashes, ts_target = video_to_phash(temp_recording_file)
            
            # 初始化所有片段的时间片段列表
            all_segments = []

            # 步骤2：遍历片段文件夹中的所有视频文件
            for clip_file in os.listdir(temp_clip_dir):
                # 构建片段视频的完整路径
                clip_file_path = os.path.join(temp_clip_dir, clip_file)
                
                # 步骤3：计算片段视频的感知哈希
                clip_file_hashes, _ = video_to_phash(clip_file_path)
                
                # 步骤4：在录播视频中查找与片段视频匹配的位置
                all_matches = match_video(clip_file_hashes, target_hashes, threshold=match_threshold)
                
                # 步骤5：过滤重叠的匹配结果，避免重复检测
                matches = filter_overlapping_matches(all_matches, min_gap=min_gap_seconds)
                
                # 步骤6：将匹配结果转换为时间片段
                segments = build_segments(matches, ts_target, len(clip_file_hashes))
                
                # 将当前片段的时间片段添加到总列表中
                all_segments.extend(segments)
                print(f"片段 {clip_file} 检测到 {len(segments)} 个匹配位置")
            
            # 合并相邻片段并导出视频
            results = merge_segments(all_segments, merge_gap_seconds)
            segments_for_export = [(s["start"], s["end"]) for s in results if isinstance(s, dict) and "start" in s and "end" in s]
            success = export_video_without_segments(recording_file, segments_for_export, output_dir)
            
            # 清理临时文件夹
            if temp_dir:
                cleanup_temp_folder(temp_dir)
            
            if success:
                return {
                    'game_name': os.path.basename(recording_file),
                    'target_video': os.path.join(output_dir, os.path.basename(recording_file)),
                    'segments_count': len(segments_for_export)
                }
            else:
                return {
                    'game_name': os.path.basename(recording_file),
                    'target_video': None,
                    'segments_count': len(segments_for_export)
                }
            
        except Exception as e:
            print(f"处理视频时发生错误: {str(e)}")
            # 确保在异常情况下也清理临时文件夹
            if temp_dir:
                cleanup_temp_folder(temp_dir)
            return None



# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoHashCutNode": VideoHashCutNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoHashCutNode": "Video Hash Cut"
}

