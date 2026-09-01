python -u run.py \
    --forecast_only \
    --device cuda:0 \
    --worker 0 \
    --cpu_worker 0 \
    --dataset_name radar \
    --dataset_path ../data/dataset/mrms/large_figure \
    --pretrained_model ../data/checkpoints/mrms_model.ckpt \
    --gen_frm_dir ./results/forecast_only \
    --num_save_samples 10 \
    --model_name NowcastNet \
    --img_height 1024 \
    --img_width 1024 \
    --case_type large \
    --img_ch 2 \
    --input_length 20 \
    --total_length 50 \
    --batch_size 1

