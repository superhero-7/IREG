# The name of experiment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
name=REG-MM-RL-REC
dataset=$2
split=$3
# mmi_margin=$4

output=ckpt/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=25556 \
    src/reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.15 \
        --clip_grad_norm 5 \
        --lr 1e-5 \
        --epochs 41 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load /data/codebase/ireg/ckpt/refcoco+/REG-MM-test/0.0001/BEST \
        --num_beams 5 \
        --batch_size 48 \
        --valid_batch_size 32 \
        --dataset $dataset\
        --dataset_split $split\
        --experiment_name $name\
        --hyperparameter_search\
        --mode 'train' \
        --ofa_ckpt_dir /data/database/REGDATA/ofa_ckpt/ \
        --refcoco_dir /data/database/REGDATA/RefCOCO \
        --img_dir /data/database/REGDATA/train2014 \
        --rl_training \
        --use_rec \
        --use_combine\
        # --mmi_margin $mmi_margin \
        # --no_evaluate \
        # --test_threshold 0.5 \
        # --dialog_training\
        # --combine_with_celoss\
        # --rl_training\
        # --debug\
        # --use_mmi \
