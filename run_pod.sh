#!/bin/bash
# 运行PODNet持续学习方法的脚本

python main_cl_jhu_pod.py \
    --datasets "fog,snow,stadium,street" \
    --model_type "final" \
    --epochs_per_task 10 \
    --lr 1e-4 \
    --config "configs/jhu_domains_cl_config.yml" \
    --use_clearml \
    --clearml_project "MPCount" \
    --clearml_task "JHU_ContinualLearning_POD" 