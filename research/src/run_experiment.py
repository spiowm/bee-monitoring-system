import gc
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import dagshub
import mlflow
from mlflow import MlflowClient
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from ultralytics import YOLO


def _setup_env(original_cwd: str) -> tuple:
    env_path = os.path.join(original_cwd, ".env")
    load_dotenv(env_path if os.path.exists(env_path) else None)

    dagshub_user  = os.getenv("DAGSHUB_USER")
    dagshub_repo  = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    return dagshub_user, dagshub_repo


def _init_dagshub(dagshub_user: str, dagshub_repo: str) -> None:
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if token:
        dagshub.auth.add_app_token(token)

    if dagshub_user and dagshub_repo:
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
    else:
        print("WARNING: DAGSHUB_USER або DAGSHUB_REPO не задані — використовується локальний MLflow.")
        dagshub.init(mlflow=True)


def _compute_pose_metrics(model, prepared_dir: Path, imgsz: int) -> dict:
    """NME and angular error for bee pose (head→stinger) on the validation set.

    NME (Normalized Mean Error) = mean keypoint distance / GT bbox diagonal.
    Angular error = circular difference between GT and predicted head→stinger angle.
    Both use IoU-based GT↔prediction matching with pool removal to prevent double-matching.

    Called on a freshly loaded model (after training model is deleted from GPU).
    """
    val_img_dir = prepared_dir / "val" / "images"
    val_lbl_dir = prepared_dir / "val" / "labels"

    img_paths = sorted(val_img_dir.glob("*.jpg"))
    if not img_paths:
        return {}

    # Use device="cpu" to guarantee no CUDA OOM after heavy 1920px training
    all_results = model.predict(
        [str(p) for p in img_paths], imgsz=imgsz, batch=1, verbose=False, device="cpu"
    )

    nme_vals, angle_errors = [], []

    for img_path, res in zip(img_paths, all_results):
        lbl_path = val_lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        gt_boxes, gt_kpts = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                if len(parts) < 9:
                    continue
                cx, cy, w, h = parts[1:5]
                gt_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                gt_kpts.append(np.array(parts[5:9]).reshape(2, 2))

        if not gt_boxes or res.boxes is None or res.keypoints is None:
            continue

        pred_boxes = res.boxes.xyxyn.cpu().numpy()    # (N, 4)
        pred_kpts  = res.keypoints.xyn.cpu().numpy()  # (N, 2, 2)
        gt_boxes_arr = np.array(gt_boxes)

        available = list(range(len(pred_boxes)))

        for gb, gk in zip(gt_boxes_arr, gt_kpts):
            if not available:
                break

            pb = pred_boxes[available]
            inter_x1 = np.maximum(gb[0], pb[:, 0])
            inter_y1 = np.maximum(gb[1], pb[:, 1])
            inter_x2 = np.minimum(gb[2], pb[:, 2])
            inter_y2 = np.minimum(gb[3], pb[:, 3])
            inter    = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            gt_area   = (gb[2] - gb[0]) * (gb[3] - gb[1])
            pred_area = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
            iou      = inter / (gt_area + pred_area - inter + 1e-6)
            best_idx = int(np.argmax(iou))
            if iou[best_idx] < 0.5:
                continue

            pred_idx = available.pop(best_idx)  # remove from pool — no double-matching
            pk = pred_kpts[pred_idx]            # (2, 2)

            # skip if any keypoint is at origin (undetected by model)
            if np.any(np.all(pk == 0, axis=1)):
                continue

            # NME: mean keypoint distance / GT bbox diagonal (both in normalized coords)
            bbox_diag = np.sqrt((gb[2] - gb[0]) ** 2 + (gb[3] - gb[1]) ** 2)
            if bbox_diag < 1e-6:
                continue
            nme_vals.append(float(np.mean(np.linalg.norm(pk - gk, axis=1)) / bbox_diag))

            # Angular error: circular difference of head→stinger vectors
            gt_vec   = gk[0] - gk[1]
            pred_vec = pk[0] - pk[1]
            if np.linalg.norm(gt_vec) < 1e-6 or np.linalg.norm(pred_vec) < 1e-6:
                continue
            diff = abs(np.arctan2(gt_vec[1], gt_vec[0]) - np.arctan2(pred_vec[1], pred_vec[0]))
            angle_errors.append(np.degrees(min(diff, 2 * np.pi - diff)))

    if not nme_vals:
        return {}

    nme  = np.array(nme_vals)
    angl = np.array(angle_errors)
    return {
        "pose_nme_mean":                 round(float(np.mean(nme)), 4),
        "pose_nme_median":               round(float(np.median(nme)), 4),
        "pose_n_matched":                len(nme_vals),
        "angular_error_mean_deg":        round(float(np.mean(angl)), 2),
        "angular_error_median_deg":      round(float(np.median(angl)), 2),
        "angular_accuracy_within_15deg": round(float(np.mean(angl < 15) * 100), 1),
        "angular_accuracy_within_30deg": round(float(np.mean(angl < 30) * 100), 1),
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    dagshub_user, dagshub_repo = _setup_env(original_cwd)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    for key in ("prepared_dir", "dataset_path"):
        if config_dict["data"].get(key):
            config_dict["data"][key] = os.path.join(original_cwd, config_dict["data"][key])

    dataset_path = config_dict["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"data.yaml не знайдено: {dataset_path}\n"
            f"Спочатку: uv run src/prepare.py experiment=<назва>"
        )

    _init_dagshub(dagshub_user, dagshub_repo)

    print(f"INFO: Модель={cfg.model.name} | Epochs={cfg.training.epochs} | imgsz={cfg.training.imgsz}")

    run_name = f"{cfg.project.experiment_name}_{datetime.now().strftime('%m%d_%H%M')}"
    mlflow.set_experiment(cfg.project.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        print(f"INFO: MLflow Run ID: {run.info.run_id}")
        mlflow.set_tag("note", cfg.project.note)
        mlflow.log_dict(config_dict, "hydra_config.yaml")

        # --- Параметри моделі ---
        mlflow.log_param("model", cfg.model.name)

        # --- Гіперпараметри навчання ---
        mlflow.log_params({
            "epochs":    cfg.training.epochs,
            "batch":     cfg.training.batch,
            "imgsz":     cfg.training.imgsz,
            "lr0":       cfg.training.lr0,
            "optimizer": cfg.training.optimizer,
            "patience":  cfg.training.patience,
            "seed":      cfg.training.seed,
            "mosaic":    cfg.training.mosaic,
            "degrees":   cfg.training.degrees,
            "fliplr":    cfg.training.fliplr,
            "translate": cfg.training.translate,
            "scale":     cfg.training.scale,
        })

        # --- Дані ---
        split_strategy = cfg.data.split_strategy
        mlflow.log_params({
            "nc":            cfg.data.nc,
            "classes":       ",".join(cfg.data.names),
            "split_strategy": split_strategy,
        })
        if split_strategy == "hive":
            mlflow.log_param("val_hives", ",".join(cfg.data.val_hives))
        else:
            mlflow.log_param("val_ratio", cfg.data.val_ratio)

        prepared_dir = Path(original_cwd) / cfg.data.prepared_dir
        train_count = len(list((prepared_dir / "train" / "images").glob("*.jpg")))
        val_count = len(list((prepared_dir / "val" / "images").glob("*.jpg")))
        mlflow.log_params({
            "train_images": train_count,
            "val_images":   val_count,
            "total_images": train_count + val_count,
        })

        # Вимикаємо вбудований MLflow-callback YOLO, щоб він не зламав наш run
        from ultralytics import settings as yolo_settings
        yolo_settings.update({"mlflow": False})
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"

        run_id = run.info.run_id

        model = YOLO(cfg.model.name)
        model.train(
            data=dataset_path,
            epochs=cfg.training.epochs,
            batch=cfg.training.batch,
            imgsz=cfg.training.imgsz,
            seed=cfg.training.seed,
            device=cfg.training.device if cfg.training.device else None,
            optimizer=cfg.training.optimizer,
            lr0=cfg.training.lr0,
            patience=cfg.training.patience,
            workers=cfg.training.workers,
            save_period=cfg.training.save_period,
            mosaic=cfg.training.mosaic,
            degrees=cfg.training.degrees,
            fliplr=cfg.training.fliplr,
            translate=cfg.training.translate,
            scale=cfg.training.scale,
            project=os.path.join(original_cwd, cfg.training.project),
            name=cfg.training.name,
        )

        # Після model.train() YOLO міг закрити/зламати активний run.
        # Відновлюємо контекст, щоб усі подальші log_metric працювали.
        client = MlflowClient()
        run_status = client.get_run(run_id).info.status
        if run_status == "FINISHED":
            client.set_terminated(run_id, status="RUNNING")

        # --- Метрики останньої епохи (train losses + val) ---
        for k, v in model.trainer.metrics.items():
            clean = k.replace("/", "_").replace("(", "_").replace(")", "")
            client.log_metric(run_id, f"last_{clean}", float(v))

        # --- Метрики best.pt ---
        best_results = model.val(
            data=dataset_path,
            project=os.path.join(original_cwd, cfg.training.project),
            name="val_best",
        )
        for k, v in best_results.results_dict.items():
            clean = k.replace("/", "_").replace("(", "_").replace(")", "")
            client.log_metric(run_id, f"best_{clean}", float(v))

        # --- Швидкість inference та пристрій ---
        speed = best_results.speed
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        total_ms = sum(speed.values())
        client.log_param(run_id, "speed_device", device_name)
        speed_metrics = {
            "speed_preprocess_ms":  round(speed.get("preprocess", 0), 3),
            "speed_inference_ms":   round(speed.get("inference", 0), 3),
            "speed_postprocess_ms": round(speed.get("postprocess", 0), 3),
            "speed_total_ms":       round(total_ms, 3),
            "speed_fps":            round(1000.0 / total_ms, 1) if total_ms > 0 else 0,
        }
        for mk, mv in speed_metrics.items():
            client.log_metric(run_id, mk, mv)

        # --- Зберігаємо шлях ---
        save_path = Path(model.trainer.save_dir)
        best_pt = save_path / "weights" / "best.pt"

        # --- Кутова точність (лише для bee_pose: 2 кточки без visibility) ---
        kpt_shape = list(cfg.data.get("kpt_shape", []))
        if kpt_shape == [2, 2]:
            print("INFO: Обчислення pose-метрик на val-сеті (на CPU для уникнення OOM)...")
            pose = _compute_pose_metrics(model, prepared_dir, imgsz=cfg.training.imgsz)
            if pose:
                for mk, mv in pose.items():
                    client.log_metric(run_id, mk, mv)
                print(f"INFO: NME={pose['pose_nme_mean']:.4f} | "
                      f"angular mean={pose['angular_error_mean_deg']}° | "
                      f"within 15°={pose['angular_accuracy_within_15deg']}% "
                      f"(n={pose['pose_n_matched']})")
            else:
                print("WARNING: Pose-метрики не вдалось обчислити (немає зіставлених пар).")

        for pattern, dest in [
            ("weights/best.pt", "yolo_run/weights"),
            ("weights/last.pt", "yolo_run/weights"),
            ("results.csv",     "yolo_run"),
            ("*.png",           "yolo_run"),
        ]:
            for f in save_path.glob(pattern):
                client.log_artifact(run_id, str(f), artifact_path=dest)

        # --- Розмір моделі ---
        if best_pt.exists():
            client.log_metric(run_id, "model_size_mb", round(best_pt.stat().st_size / 1e6, 2))

        print(f"INFO: Артефакти збережено з {save_path}")

    print("INFO: Завершено. Всі артефакти збережено в DagsHub/MLflow.")


if __name__ == "__main__":
    main()
