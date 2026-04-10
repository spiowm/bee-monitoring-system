import subprocess
import os

def ensure_dataset(config: dict):
    """
    Завантажує датасет з Kaggle або перевіряє його наявність локально.
    """
    dataset_config = config.get("data", {})
    source_type = dataset_config.get("source_type", "kaggle")
    kaggle_id = dataset_config.get("kaggle_id", "")
    kaggle_file = dataset_config.get("kaggle_file", "") # завантаження конкретного файлу
    target_dir = dataset_config.get("target_dir", "datasets/kaggle_dataset")
    dataset_yaml = dataset_config.get("dataset_path", "")

    if source_type == "kaggle" and kaggle_id:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        if dataset_yaml and os.path.exists(dataset_yaml):
            print(f"INFO: Kaggle датасет вже завантажено і знайдено за шляхом: {dataset_yaml}")
            return

        file_msg = f"окремий архів {kaggle_file}" if kaggle_file else "весь датасет"
        print(f"INFO: Завантаження ({file_msg}) з Kaggle ({kaggle_id})...")
        print("INFO: Переконайтесь, що у вас встановлено Kaggle API token.")
        
        # Виклик kaggle api CLI
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", target_dir, "--unzip"]
        if kaggle_file:
            cmd.extend(["-f", kaggle_file])
            
        try:
            result = subprocess.run(cmd, check=True)
            print(f"INFO: Kaggle дані успішно завантажено та розпаковано у {target_dir}")
        except subprocess.CalledProcessError as e:
            print("ERROR: Помилка завантаження датасету з Kaggle. Перевірте ваші ключі та ID датасету.")
            return

    # Фінальна перевірка yaml
    if dataset_yaml and not os.path.exists(dataset_yaml):
        print(f"WARNING: Файл конфігурації data.yaml не знайдено за шляхом: {dataset_yaml}")
    else:
        print(f"INFO: Датасет готовий: {dataset_yaml}")
