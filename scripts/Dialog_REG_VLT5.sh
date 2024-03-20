# The name of experiment
export CUDA_VISIBLE_DEVICES=$4
name=REG-MM-Dialog-FromEpoch30-onlyDialogloss
dataset=$2
split=$3

output=ckpt/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=$8 \
    src/multitask_reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-6 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load $6 \
        --num_beams 5 \
        --batch_size 256 \
        --valid_batch_size 1 \
        --dataset $dataset \
        --dataset_split $split\
        --experiment_name $name\
        --hyperparameter_search\
        --zero_shot_test\
        --use_rec\
        --dialog_round 2 \
        --last_round \
        --bad_res_path $7 \
        --test_threshold 0.5 \
        --mode 'train' \
        --dialog_sp_training\
        --use_detector \
        --ofa_ckpt_dir /data/database/REGDATA/ofa_ckpt/ \
        --refcoco_dir /data/database/REGDATA/RefCOCO \
        --img_dir /data/database/REGDATA/train2014 \
        --base_load $5 \
        --only_dialog_loss \
        # --debug\
        # --no_evaluate \
        # --refine \
        # --combine_with_celoss\
        # --use_combine\
        # --rl_training\
        # --use_mmi \

