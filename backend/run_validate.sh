#!/bin/bash
# Валідація РЕАЛЬНИМ пайплайном (eval_cli, некешований YOLO) — підтвердження,
# що кеш-метрики тримаються в продакшені. Читає config/eval_config.yaml.
cd /home/spiowm/spi/projects/bee-monitoring-system/backend || exit 1
for p in 20230609b-def 20230711a-fan 20230711b-fan; do
  echo "######## REAL PIPELINE: $p ########"
  uv run python eval_cli.py --pair $p --json /tmp/real_$p.json 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
done
echo "VALIDATE DONE"
