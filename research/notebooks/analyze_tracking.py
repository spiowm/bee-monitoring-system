#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict

def get_video_info(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        return {'fps': fps, 'frames': frame_count, 'width': width, 'height': height, 'duration': duration}
    except Exception:
        # fallback to ffprobe
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            info = json.loads(result.stdout)
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video':
                    width = int(stream['width'])
                    height = int(stream['height'])
                    frames = int(stream.get('nb_frames', 0))
                    duration = float(stream.get('duration', 0))
                    fps_str = stream.get('r_frame_rate', '0/1')
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den > 0 else 0
                    if frames == 0 and duration > 0 and fps > 0:
                        frames = int(duration * fps)
                    return {'fps': fps, 'frames': frames, 'width': width, 'height': height, 'duration': duration}
        except:
            pass
    return None

def analyze_behavior_and_tracks(file_path):
    tracks = defaultdict(list)
    stats = {'for': 0, 'def': 0, 'fan': 0, 'wash': 0, 'rows': 0, 'multiple': 0}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 1:
                parts = line.strip().split()
                
            if len(parts) < 2:
                continue
                
            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                tracks[track_id].append(frame_id)
                stats['rows'] += 1
                
                # Analyze behavior if columns exist
                if len(parts) >= 10:
                    c_for = int(float(parts[6]))
                    c_def = int(float(parts[7]))
                    c_fan = int(float(parts[8]))
                    c_wash = int(float(parts[9]))
                    
                    if c_for: stats['for'] += 1
                    if c_def: stats['def'] += 1
                    if c_fan: stats['fan'] += 1
                    if c_wash: stats['wash'] += 1
                    
                    if sum([c_for, c_def, c_fan, c_wash]) > 1:
                        stats['multiple'] += 1
            except ValueError:
                pass

    num_tracks = len(tracks)
    track_lengths = [len(frames) for frames in tracks.values()]
    avg_len = sum(track_lengths) / num_tracks if num_tracks > 0 else 0
    max_len = max(track_lengths) if track_lengths else 0
    min_len = min(track_lengths) if track_lengths else 0
    
    stats.update({
        'unique_tracks': num_tracks,
        'avg_track_len': avg_len,
        'max_track_len': max_len,
        'min_track_len': min_len
    })
    return stats

def main():
    base_dir = Path("/home/spiowm/spi/projects/bee-monitoring-system/research/datasets/raw/tracking_and_behavior")
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print("=========================================================")
    print("КОМПЛЕКСНИЙ АНАЛІЗ ДАТАСЕТУ tracking_and_behavior")
    print("=========================================================")
    print(f"Знайдено відео (директорій): {len(subdirs)}\n")
    
    total_class_frames = {'for': 0, 'def': 0, 'fan': 0, 'wash': 0}
    total_multiple_flags = 0
    
    for d in subdirs:
        print(f"--- Папка: {d.name} ---")
        
        # 1. Video Info
        vid_path = d / "video.mp4"
        if vid_path.exists():
            vinfo = get_video_info(vid_path)
            if vinfo:
                print(f"[Відео]: {vinfo['width']}x{vinfo['height']}, " 
                      f"Кадрів: {vinfo['frames']}, FPS: {vinfo['fps']:.2f}, "
                      f"Тривалість: {vinfo['duration']:.2f} сек.")
            else:
                print("[Відео]: video.mp4 присутнє (метадані недоступні).")
        else:
            print("[Відео]: video.mp4 ВІДСУТНЄ!")
            
        # 2. Entrance Zone
        ez_path = d / "entrance_zone.txt"
        if ez_path.exists():
            with open(ez_path, 'r') as f:
                print(f"[Зона льотка]: {f.read().strip()}")
        else:
            print("[Зона льотка]: ВІДСУТНІЙ!")
            
        # 3. Tracks & Behavior
        beh_path = d / "tracks_and_behavior.txt"
        if beh_path.exists():
            b_info = analyze_behavior_and_tracks(beh_path)
            print(f"[Треки]: Рядків (bbox): {b_info['rows']}, Унікальних бджіл: {b_info['unique_tracks']}")
            print(f"[Життєвий цикл]: Середня тривалість треку: {b_info['avg_track_len']:.1f} кадрів (Мін: {b_info['min_track_len']}, Макс: {b_info['max_track_len']})")
            print(f"[Поведінка]: Foraging: {b_info['for']}, Defense: {b_info['def']}, Fanning: {b_info['fan']}, Washboarding: {b_info['wash']}")
            
            # Aggregate totals
            total_class_frames['for'] += b_info['for']
            total_class_frames['def'] += b_info['def']
            total_class_frames['fan'] += b_info['fan']
            total_class_frames['wash'] += b_info['wash']
            total_multiple_flags += b_info['multiple']
        else:
            print("[tracks_and_behavior.txt]: ВІДСУТНІЙ!")
            
        print() # empty line
        
    print("=========================================================")
    print("ЗАГАЛЬНА СТАТИСТИКА (по всім відео)")
    print("=========================================================")
    print(f"Foraging: {total_class_frames['for']} bbox")
    print(f"Defense: {total_class_frames['def']} bbox")
    print(f"Fanning: {total_class_frames['fan']} bbox")
    print(f"Washboarding: {total_class_frames['wash']} bbox")
    
    if total_multiple_flags > 0:
        print(f"\n[УВАГА]: Знайдено {total_multiple_flags} кадрів, де бджола виконує кілька дій одночасно.")
    else:
        print("\n[INFO]: Класи повністю взаємовиключні (одна дія на кадр).")

if __name__ == "__main__":
    main()
