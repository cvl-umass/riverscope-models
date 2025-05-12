#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=[l40s|a100]
#SBATCH --nodes=1 
#SBATCH -c 16  # Number of Cores per Task
#SBATCH -G 1 # Number of GPUs
#SBATCH --mem=60G
#SBATCH -o slurm_train_planet/water-all_combinations-%j-%a.out  # %j = job ID
#SBATCH --job-name=planet-water-all_combinations
#SBATCH --time=04:00:00 
#SBATCH --mail-type=ALL
#SBATCH --array=0-26    # for linear adapter 
# #SBATCH --array=0-66    # for all adapter types
# #SBATCH --array=27-40    # for random inititalization (no_init)
# #SBATCH --array=41-66    # for dropping the last channel (drop)

lrs=(0.000100 0.001000 0.000100 0.001000 0.000100 0.000100 0.000100 0.000100 0.001000 0.001000 0.000100 0.000100 0.000100 0.001000 0.001000 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000010 0.000100 0.000100 0.000010 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.001000 0.000100 0.000100 0.001000 0.001000 0.000100 0.000100 0.000010 0.000100 0.000100 0.000010 0.001000 0.001000 0.000100 0.000100 0.000100 0.000100 0.000100 0.001000 0.000100 0.000100 0.000100 0.001000 0.000100 0.001000 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000100 0.000010)
segment_models=("unet" "unet" "unet" "unet" "unet" "unet" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "dpt" "dpt" "dpt" "dpt" "dpt" "dpt" "unet" "unet" "unet" "unet" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "fpn" "fpn" "fpn" "fpn" "dpt" "dpt" "unet" "unet" "unet" "unet" "unet" "unet" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "deeplabv3" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "fpn" "dpt" "dpt" "dpt" "dpt" "dpt")
adaptors=("linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "no_init" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop" "drop")
backbones=("resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "satlas_si_swinb" "satlas_si_swint" "satlas_si_resnet50" "vitl" "vitb" "vitb_dino" "vitb_mocov3" "vitb_clip" "vitb_prithvi" "resnet50" "mobilenet_v2" "swint" "swinb" "resnet50" "mobilenet_v2" "swint" "swinb" "resnet50" "mobilenet_v2" "swint" "swinb" "vitl" "vitb" "resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "resnet50" "resnet50_seco" "resnet50_mocov3" "mobilenet_v2" "swint" "swinb" "satlas_si_swinb" "satlas_si_swint" "satlas_si_resnet50" "vitl" "vitb" "vitb_dino" "vitb_mocov3" "vitb_clip")
heads=("no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "satlas_head" "satlas_head" "satlas_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "no_head" "satlas_head" "satlas_head" "satlas_head" "no_head" "no_head" "no_head" "no_head" "no_head")
sizes=(512 512 512 512 224 224 512 512 512 512 224 224 512 512 512 512 224 224 512 512 512 224 224 224 224 224 224 512 512 224 224 512 512 224 224 512 512 224 224 224 224 512 512 512 512 224 224 512 512 512 512 224 224 512 512 512 512 224 224 512 512 512 224 224 224 224 224)
urls=("tcp://127.0.0.1:8000" "tcp://127.0.0.1:8001" "tcp://127.0.0.1:8002" "tcp://127.0.0.1:8003" "tcp://127.0.0.1:8004" "tcp://127.0.0.1:8005" "tcp://127.0.0.1:8006" "tcp://127.0.0.1:8007" "tcp://127.0.0.1:8008" "tcp://127.0.0.1:8009" "tcp://127.0.0.1:8010" "tcp://127.0.0.1:8011" "tcp://127.0.0.1:8012" "tcp://127.0.0.1:8013" "tcp://127.0.0.1:8014" "tcp://127.0.0.1:8015" "tcp://127.0.0.1:8016" "tcp://127.0.0.1:8017" "tcp://127.0.0.1:8018" "tcp://127.0.0.1:8019" "tcp://127.0.0.1:8020" "tcp://127.0.0.1:8021" "tcp://127.0.0.1:8022" "tcp://127.0.0.1:8023" "tcp://127.0.0.1:8024" "tcp://127.0.0.1:8025" "tcp://127.0.0.1:8026" "tcp://127.0.0.1:8027" "tcp://127.0.0.1:8028" "tcp://127.0.0.1:8029" "tcp://127.0.0.1:8030" "tcp://127.0.0.1:8031" "tcp://127.0.0.1:8032" "tcp://127.0.0.1:8033" "tcp://127.0.0.1:8034" "tcp://127.0.0.1:8035" "tcp://127.0.0.1:8036" "tcp://127.0.0.1:8037" "tcp://127.0.0.1:8038" "tcp://127.0.0.1:8039" "tcp://127.0.0.1:8040" "tcp://127.0.0.1:8041" "tcp://127.0.0.1:8042" "tcp://127.0.0.1:8043" "tcp://127.0.0.1:8044" "tcp://127.0.0.1:8045" "tcp://127.0.0.1:8046" "tcp://127.0.0.1:8047" "tcp://127.0.0.1:8048" "tcp://127.0.0.1:8049" "tcp://127.0.0.1:8050" "tcp://127.0.0.1:8051" "tcp://127.0.0.1:8052" "tcp://127.0.0.1:8053" "tcp://127.0.0.1:8054" "tcp://127.0.0.1:8055" "tcp://127.0.0.1:8056" "tcp://127.0.0.1:8057" "tcp://127.0.0.1:8058" "tcp://127.0.0.1:8059" "tcp://127.0.0.1:8060" "tcp://127.0.0.1:8061" "tcp://127.0.0.1:8062" "tcp://127.0.0.1:8063" "tcp://127.0.0.1:8064" "tcp://127.0.0.1:8065" "tcp://127.0.0.1:8066")

# conda_segment
# Combinations:
# [unet, deeplabv3] * [resnet50, resnet50_seco, resnet50_mocov3, swint, swinb]
# fpn * [satlas_swint, satlas_swinb, satlas_resnet50, resnet50, resnet50_seco, resnet50_mocov3, swint, swinb]
echo train_planet.py \
    --lr ${lrs[$SLURM_ARRAY_TASK_ID]} \
    --segment_model ${segment_models[$SLURM_ARRAY_TASK_ID]} \
    --adaptor ${adaptors[$SLURM_ARRAY_TASK_ID]} \
    --backbone ${backbones[$SLURM_ARRAY_TASK_ID]} \
    --head ${heads[$SLURM_ARRAY_TASK_ID]} \
    --resize_size ${sizes[$SLURM_ARRAY_TASK_ID]} \
    --batch_size 32 \
    --dist-url ${urls[$SLURM_ARRAY_TASK_ID]} \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --loss_type 'bce'

python train_planet.py \
    --lr ${lrs[$SLURM_ARRAY_TASK_ID]} \
    --segment_model ${segment_models[$SLURM_ARRAY_TASK_ID]} \
    --adaptor ${adaptors[$SLURM_ARRAY_TASK_ID]} \
    --backbone ${backbones[$SLURM_ARRAY_TASK_ID]} \
    --head ${heads[$SLURM_ARRAY_TASK_ID]} \
    --resize_size ${sizes[$SLURM_ARRAY_TASK_ID]} \
    --batch_size 32 \
    --dist-url ${urls[$SLURM_ARRAY_TASK_ID]} \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --out "./results/planet-water-best-perf" \
    --loss_type 'bce'

