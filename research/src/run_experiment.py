import os
import dagshub
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from ultralytics import YOLO
from data_manager import ensure_dataset

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Головний пайплайн навчання, який використовує Hydra для керування конфігурацією.
    """
    # 1. Завантаження змінних оточення (з файлу .env у корені research)
    # Змінюємо директорію, бо Hydra динамічно створює піддиректорії (наприклад outputs/YYYY-MM-DD)
    original_cwd = hydra.utils.get_original_cwd()
    env_path = os.path.join(original_cwd, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_token:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    
    # Перетворюємо DictConfig у звичайний словник для data_manager
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Підготовка даних (Kaggle pull)
    dataset_target_dir = os.path.join(original_cwd, config_dict["data"].get("target_dir", "datasets/kaggle_dataset"))
    config_dict["data"]["target_dir"] = dataset_target_dir
    if config_dict["data"].get("dataset_path"):
        config_dict["data"]["dataset_path"] = os.path.join(original_cwd, config_dict["data"]["dataset_path"])
        
    ensure_dataset(config_dict)
    
    # 3. Ініціалізація DagsHub
    experiment_name = cfg.project.experiment_name
    note = cfg.project.note
    
    if dagshub_user and dagshub_repo:
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
    else:
        print("WARNING: В .env не знайдено DAGSHUB_USER або DAGSHUB_REPO. Ініціалізація DagsHub може бути неповною.")
        dagshub.init(mlflow=True)  
        
    mlflow.set_experiment(experiment_name)
    
    # 4. Ініціалізація моделі
    model_name = cfg.model.name
    task = cfg.model.task
    print(f"INFO: Ініціалізація моделі {model_name} для задачі: {task}")
    
    model_path = model_name  
    model = YOLO(model_path)
    
    # 5. Запуск експерименту в контексті MLflow
    with mlflow.start_run() as run:
        print(f"INFO: Початок MLflow Run: {run.info.run_id}")
        
        mlflow.set_tag("note", note)
        mlflow.log_dict(config_dict, "hydra_config.yaml") 
        
        model.train(
            data=config_dict["data"].get("dataset_path"),
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
            name=cfg.training.name
        )
        
        print("INFO: Тренування завершено! Всі метрики та артефакти завантажено в DagsHub/MLflow.")

if __name__ == "__main__":
    main()
