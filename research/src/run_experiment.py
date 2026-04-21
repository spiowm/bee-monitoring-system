import os
from datetime import datetime
from pathlib import Path
import dagshub
import mlflow
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
        # --- Метрики останньої епохи (train losses + val) ---
        for k, v in model.trainer.metrics.items():
            clean = k.replace("/", "_").replace("(", "_").replace(")", "")
            mlflow.log_metric(f"last_{clean}", float(v))

        # --- Метрики best.pt ---
        best_results = model.val(
            data=dataset_path,
            project=os.path.join(original_cwd, cfg.training.project),
            name="val_best",
        )
        for k, v in best_results.results_dict.items():
            clean = k.replace("/", "_").replace("(", "_").replace(")", "")
            mlflow.log_metric(f"best_{clean}", float(v))

        save_path = Path(model.trainer.save_dir)
        for pattern, dest in [
            ("weights/best.pt", "yolo_run/weights"),
            ("weights/last.pt", "yolo_run/weights"),
            ("results.csv",     "yolo_run"),
            ("*.png",           "yolo_run"),
        ]:
            for f in save_path.glob(pattern):
                mlflow.log_artifact(str(f), artifact_path=dest)
        print(f"INFO: Артефакти збережено з {save_path}")

    print("INFO: Завершено. Всі артефакти збережено в DagsHub/MLflow.")


if __name__ == "__main__":
    main()
