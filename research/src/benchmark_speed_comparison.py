import os
import time
import cv2
import torch
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def get_bee_count(label_file):
    if not os.path.exists(label_file):
        return 0
    with open(label_file, 'r') as f:
        return sum(1 for line in f if line.strip())

def group_images(images_dir, labels_dir):
    groups = {
        "2-3": (2, 3),
        "4-6": (4, 6),
        "7-9": (7, 9),
        "10-12": (10, 12),
        "13-15": (13, 15),
        "16-20": (16, 20)
    }
    
    grouped_files = {k: [] for k in groups.keys()}
    
    image_files = list(Path(images_dir).glob("*.jpg"))
    for img_path in image_files:
        label_file = Path(labels_dir) / f"{img_path.stem}.txt"
        count = get_bee_count(str(label_file))
        
        for g_name, (g_min, g_max) in groups.items():
            if g_min <= count <= g_max:
                if len(grouped_files[g_name]) < 10:
                    grouped_files[g_name].append((str(img_path), count))
                break
    return grouped_files

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    model_path = project_root / "backend" / "data" / "models" / "bee_pose" / "best.pt"
    images_dir = project_root / "research" / "datasets" / "raw" / "pose" / "images"
    labels_dir = project_root / "research" / "datasets" / "raw" / "pose" / "labels"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = YOLO(str(model_path))
    
    cache_file = project_root / "research" / "benchmark_selected_images.json"
    if cache_file.exists():
        print(f"Loading selected images from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            grouped_files = json.load(f)
    else:
        print("Scanning dataset to group images by bee count...")
        grouped_files = group_images(images_dir, labels_dir)
        with open(cache_file, 'w') as f:
            json.dump(grouped_files, f, indent=4)
        print(f"Saved selected images cache to: {cache_file}")
    
    results = []
    
    warmup_runs = 10
    measure_runs = 50
    
    print("Starting benchmark...")
    
    for group_name, files in grouped_files.items():
        if not files:
            print(f"Warning: No images found for group {group_name}")
            continue
            
        for img_path, count in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            print(f"Benchmarking group {group_name}, {count} bees...")
            
            # Warmup
            for _ in range(warmup_runs):
                model.predict(img, imgsz=1920, verbose=False, device=device)
                
            # ONE STEP measurement
            one_step_times = []
            for _ in range(measure_runs):
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.predict(img, imgsz=1920, verbose=False, device=device)
                if device == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()
                one_step_times.append((t1 - t0) * 1000)
            
            avg_one_step_ms = sum(one_step_times) / measure_runs
            
            # TWO STEP measurement
            two_step_times = []
            for _ in range(measure_runs):
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                res = model.predict(img, imgsz=1920, verbose=False, device=device)
                
                if device == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()
                step1_time = (t1 - t0) * 1000
                
                bboxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
                
                crop_times = []
                for box in bboxes:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    crop_resized = cv2.resize(crop, (64, 64))
                    
                    if device == "cuda": torch.cuda.synchronize()
                    tc0 = time.perf_counter()
                    model.predict(crop_resized, imgsz=64, verbose=False, device=device)
                    if device == "cuda": torch.cuda.synchronize()
                    tc1 = time.perf_counter()
                    crop_times.append((tc1 - tc0) * 1000)
                
                avg_crop_time = sum(crop_times) / len(crop_times) if crop_times else 0
                total_two_step = step1_time + len(bboxes) * avg_crop_time
                two_step_times.append(total_two_step)
                
            avg_two_step_ms = sum(two_step_times) / measure_runs
            speedup_ratio = avg_one_step_ms / avg_two_step_ms if avg_two_step_ms > 0 else 0
            
            results.append({
                "group": group_name,
                "n_bees_actual": count,
                "one_step_ms": avg_one_step_ms,
                "two_step_ms": avg_two_step_ms,
                "speedup_ratio": speedup_ratio
            })

    # Output to CSV
    csv_file = project_root / "research" / "benchmark_speed_comparison.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n_bees_actual", "one_step_ms", "two_step_ms", "speedup_ratio"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nDetailed results saved to {csv_file}")
    
    # Average the results by group for clearer summary and plotting
    group_summaries = {}
    for r in results:
        g = r['group']
        if g not in group_summaries:
            group_summaries[g] = {'count': 0, 'actual_bees': 0, 'one_step': 0, 'two_step': 0}
        group_summaries[g]['count'] += 1
        group_summaries[g]['actual_bees'] += r['n_bees_actual']
        group_summaries[g]['one_step'] += r['one_step_ms']
        group_summaries[g]['two_step'] += r['two_step_ms']
        
    for g in group_summaries:
        c = group_summaries[g]['count']
        group_summaries[g]['actual_bees'] /= c
        group_summaries[g]['one_step'] /= c
        group_summaries[g]['two_step'] /= c

    # Print summary table
    print("\nSummary Table (Averaged per group):")
    print(f"{'Group':<10} {'Avg Bees':<10} {'One Step (ms)':<15} {'Two Step (ms)':<15} {'Speedup (1/2)':<15}")
    for g, sums in group_summaries.items():
        ratio_1_to_2 = sums['one_step'] / sums['two_step'] if sums['two_step'] > 0 else 0
        print(f"{g:<10} {sums['actual_bees']:<10.1f} {sums['one_step']:<15.2f} {sums['two_step']:<15.2f} {ratio_1_to_2:<15.2f}")
        
    # Plot results
    if not results:
        print("No results to plot.")
        return

    # Use all individual points or aggregated? The prompt says "Вісь X: кількість бджіл", so individual or avg.
    # Let's plot aggregated points.
    sorted_groups = sorted(group_summaries.items(), key=lambda x: x[1]['actual_bees'])
    bees_count_sorted = np.array([x[1]['actual_bees'] for x in sorted_groups])
    one_step_sorted = np.array([x[1]['one_step'] for x in sorted_groups])
    two_step_sorted = np.array([x[1]['two_step'] for x in sorted_groups])
    
    intersection_idx = None
    for i in range(len(bees_count_sorted)):
        if two_step_sorted[i] > one_step_sorted[i]:
            intersection_idx = i
            break
            
    plt.figure(figsize=(10, 6))
    plt.plot(bees_count_sorted, one_step_sorted, label="One-step (1920)", color="blue", marker='o')
    plt.plot(bees_count_sorted, two_step_sorted, label="Two-step (1920 + 64x64 crops)", color="red", marker='x')
    
    if intersection_idx is not None and intersection_idx > 0:
        x1, x2 = bees_count_sorted[intersection_idx-1], bees_count_sorted[intersection_idx]
        y1_1, y1_2 = one_step_sorted[intersection_idx-1], one_step_sorted[intersection_idx]
        y2_1, y2_2 = two_step_sorted[intersection_idx-1], two_step_sorted[intersection_idx]
        
        m1 = (y1_2 - y1_1) / (x2 - x1) if x2 != x1 else 0
        c1 = y1_1 - m1 * x1
        
        m2 = (y2_2 - y2_1) / (x2 - x1) if x2 != x1 else 0
        c2 = y2_1 - m2 * x1
        
        if m1 != m2:
            x_int = (c2 - c1) / (m1 - m2)
            y_int = m1 * x_int + c1
            if bees_count_sorted[0] <= x_int <= bees_count_sorted[-1]:
                plt.plot(x_int, y_int, 'go', markersize=10, label=f"Intersection (~{x_int:.1f} bees)")
    
    plt.xlabel("Number of Bees per Frame (Average)")
    plt.ylabel("Inference Time (ms)")
    plt.title("One-Step vs Simulated Two-Step Inference Speed")
    plt.legend()
    plt.grid(True)
    
    plot_file = project_root / "research" / "benchmark_speed_comparison.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    main()
