import argparse
import os
import time
import cv2
import torch
import mlflow
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="FPS Benchmark for YOLO models")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--image", type=str, default="datasets/pose/val/images/20230609e318.jpg", help="Path to val image")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Measurement iterations")
    return parser.parse_args()

def find_fallback_image():
    """Find any .jpg image in datasets/ as a fallback."""
    for path in Path("datasets").rglob("*.jpg"):
        return str(path)
    return None

def setup_mlflow():
    """Setup DagsHub/MLflow if env vars are present."""
    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_user and dagshub_repo and dagshub_token:
        try:
            import dagshub
            dagshub.auth.add_app_token(dagshub_token)
            dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
            print(f"Initialized DagsHub MLflow: {dagshub_user}/{dagshub_repo}")
        except ImportError:
            print("dagshub package not found, logging to local MLflow.")
    
    mlflow.set_experiment("FPS_Benchmark")

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")
    model = YOLO(args.model)
    
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found.")
        fallback = find_fallback_image()
        if fallback:
            print(f"Using fallback image: {fallback}")
            image_path = fallback
        else:
            raise FileNotFoundError("Could not find any image for benchmarking in datasets/")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        _ = model.predict(img, imgsz=args.imgsz, verbose=False, device=device)
        
    print(f"Running {args.iters} measurement iterations...")
    
    total_time_ms = 0
    prep_ms, inf_ms, post_ms = 0.0, 0.0, 0.0
    
    for _ in range(args.iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        results = model.predict(img, imgsz=args.imgsz, verbose=False, device=device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        total_time_ms += (t1 - t0) * 1000
        speeds = results[0].speed
        prep_ms += speeds.get("preprocess", 0)
        inf_ms += speeds.get("inference", 0)
        post_ms += speeds.get("postprocess", 0)
        
    avg_total = total_time_ms / args.iters
    avg_prep = prep_ms / args.iters
    avg_inf = inf_ms / args.iters
    avg_post = post_ms / args.iters
    fps_real = 1000.0 / avg_total
    
    print("\n" + "="*40)
    print(f"{'FPS Benchmark Summary':^40}")
    print("="*40)
    print(f"Model:       {args.model}")
    print(f"Image size:  {args.imgsz}")
    print(f"Device:      {device}")
    print("-" * 40)
    print(f"Preprocess:  {avg_prep:.2f} ms")
    print(f"Inference:   {avg_inf:.2f} ms")
    print(f"Postprocess: {avg_post:.2f} ms")
    print(f"Total (E2E): {avg_total:.2f} ms")
    print(f"Real FPS:    {fps_real:.2f}")
    print("="*40 + "\n")
    
    setup_mlflow()
    with mlflow.start_run():
        mlflow.log_params({
            "model_path": args.model,
            "imgsz": args.imgsz,
            "device": device,
            "batch": 1,
            "warmup": args.warmup,
            "iters": args.iters,
            "image_used": image_path
        })
        mlflow.log_metrics({
            "preprocess_ms": avg_prep,
            "inference_ms": avg_inf,
            "postprocess_ms": avg_post,
            "total_ms": avg_total,
            "fps_real": fps_real
        })
        print("Logged results to MLflow.")

if __name__ == "__main__":
    main()