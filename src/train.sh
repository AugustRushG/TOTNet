#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=Aug_occl

nvidia-smi
export NCCL_P2P_DISABLE=1
SAVE_FOLDER='TOTNet_Tennis_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64'
NUM_FRAMES=5
BATCH_SIZE=8
MODEL_CHOICE='TOTNet'
DATASET_CHOICE='tennis'
NUM_CHANNELS=64


torchrun --standalone --nproc_per_node=1 main.py     \
    --num_epochs 30   \
    --saved_fn $SAVE_FOLDER   \
    --num_frames $NUM_FRAMES  \
    --checkpoint_freq 10 \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --loss_function WBCE  \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --num_channels $NUM_CHANNELS \
    --batch_size $BATCH_SIZE \
    --print_freq 50 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice $DATASET_CHOICE \
    --weighting_list 1 2 3 3  \
    --model_choice $MODEL_CHOICE  \
    --occluded_prob 0.25 \
    --ball_size 8 \
    --val-size 0.2 \
    --no_test   \

torchrun --nproc_per_node=1 test.py \
  --working-dir '../' \
  --saved_fn $SAVE_FOLDER \
  --model_choice $MODEL_CHOICE  \
  --gpu_idx 0   \
  --batch_size $BATCH_SIZE   \
  --img_size 288 512    \
  --num_frames $NUM_FRAMES  \
  --dataset_choice $DATASET_CHOICE \
  --num_channels $NUM_CHANNELS \
  --ball_size 4 \
  --pretrained_path "../checkpoints/$SAVE_FOLDER/${SAVE_FOLDER}_best.pth"\