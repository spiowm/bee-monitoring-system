import os
import dagshub
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_env(original_cwd: str) -> tuple:
    """Завантажує .env і повертає DagsHub credentials."""
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


def _train_model(
    cfg: DictConfig,
    dataset_path: str,
    original_cwd: str,
    run_name: str = None,
) -> tuple:
    """Навчає YOLO-модель і повертає (model, save_dir)."""
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
        project=os.path.join(original_cwd, cfg.training.project),
        name=run_name or cfg.training.name,
    )
    return model, str(model.trainer.save_dir)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_single(cfg: DictConfig, config_dict: dict, dataset_path: str, original_cwd: str) -> None:
    """Один тренувальний запуск з логуванням в MLflow."""
    mlflow.set_experiment(cfg.project.experiment_name)

    with mlflow.start_run() as run:
        print(f"INFO: MLflow Run ID: {run.info.run_id}")
        mlflow.set_tag("note", cfg.project.note)
        mlflow.log_dict(config_dict, "hydra_config.yaml")

        model, save_dir = _train_model(cfg, dataset_path, original_cwd)

        mlflow.log_artifacts(save_dir, artifact_path="yolo_run")
        print(f"INFO: Артефакти збережено з {save_dir}")


# ---------------------------------------------------------------------------
# HPO (Optuna)
# ---------------------------------------------------------------------------

def _run_trial(trial, cfg: DictConfig, dataset_path: str, original_cwd: str) -> float:
    """Один Optuna trial як вкладений MLflow run."""
    search = cfg.hpo.search_space

    lr0   = trial.suggest_float("lr0",  search.lr0[0],  search.lr0[1],  log=True)
    batch = trial.suggest_categorical("batch", list(search.batch))
    imgsz = trial.suggest_categorical("imgsz", list(search.imgsz))

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params({"lr0": lr0, "batch": batch, "imgsz": imgsz, "trial": trial.number})

        # Тимчасово перевизначаємо параметри для цього trial
        trial_cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"lr0": lr0, "batch": batch, "imgsz": imgsz}}))
        model, save_dir = _train_model(trial_cfg, dataset_path, original_cwd, run_name=f"hpo_trial_{trial.number}")

        metric_key = cfg.hpo.metric
        metric_val = float(model.trainer.metrics.get(metric_key, 0.0))
        mlflow.log_metric(metric_key.replace("/", "_"), metric_val)
        mlflow.log_artifacts(save_dir, artifact_path=f"trial_{trial.number}")

    return metric_val


def _run_hpo(cfg: DictConfig, config_dict: dict, dataset_path: str, original_cwd: str) -> None:
    """Запускає Optuna HPO, кожен trial — окремий MLflow run."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow.set_experiment(cfg.project.experiment_name + "_HPO")

    with mlflow.start_run(run_name="hpo_study") as run:
        print(f"INFO: HPO MLflow Run ID: {run.info.run_id}")
        print(f"INFO: Кількість trials: {cfg.hpo.n_trials} | Метрика: {cfg.hpo.metric}")
        mlflow.log_dict(config_dict, "hydra_config.yaml")
        mlflow.log_param("n_trials", cfg.hpo.n_trials)

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            study_name=cfg.project.experiment_name,
        )
        study.optimize(
            lambda trial: _run_trial(trial, cfg, dataset_path, original_cwd),
            n_trials=cfg.hpo.n_trials,
        )

        best = study.best_params
        mlflow.log_params({"best_" + k: v for k, v in best.items()})
        mlflow.log_metric("best_" + cfg.hpo.metric.replace("/", "_"), study.best_value)

        print(f"INFO: Найкращі параметри: {best}")
        print(f"INFO: Найкраще значення ({cfg.hpo.metric}): {study.best_value:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Головний пайплайн навчання.

    Режими:
      single (default) — один запуск за конфігом
      hpo              — Optuna HPO (mode=hpo hpo.n_trials=N)

    Hydra multirun (grid sweep) — без зміни коду:
      uv run src/run_experiment.py --multirun training.lr0=0.001,0.0005 training.imgsz=1280,1920
    """
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

    mode = str(cfg.get("mode", "single"))
    print(f"INFO: Режим={mode} | Модель={cfg.model.name} | Epochs={cfg.training.epochs} | imgsz={cfg.training.imgsz}")

    if mode == "hpo":
        _run_hpo(cfg, config_dict, dataset_path, original_cwd)
    else:
        _run_single(cfg, config_dict, dataset_path, original_cwd)

    print("INFO: Завершено. Всі артефакти збережено в DagsHub/MLflow.")


if __name__ == "__main__":
    main()
