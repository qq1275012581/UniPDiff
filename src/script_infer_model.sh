# !/bin/bash

python infer_model.py \
    --ckpt_pth "/home-ssd/Users/gm_intern/liguowen/UniPDiff/model_params/UniPDiff_Params.pth" \
    --output_dir "/home-ssd/Users/gm_intern/liguowen/py-ResCast/v3_pangu/tp_output/" \
    --model_name "pangu" \
    --start_time "2022010200" \
    --end_time "2022123100" \
    --idx_of_day 1 3 \
    --sample_steps 100 \
    # --save_tp \
    # --visualize \
    # --ensemble_forecast
    