#!/bin/bash  




DEVICE=cuda
NUM_GPUS=1
MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_addr $MY_MASTER_ADDR --master_port $MY_MASTER_PORT --use_env \
evaluation.py \
--data_folder ../data \
--eval_downsamp_t 2 \
--eval_dataset Evaluation_rb2d_uniform_ra1e5_dt_0_1_t_50_60s_S_300.npz \
--eval_downsamp_xz 4 \
--checkpoint ../train/testing_dataLoader_velOnly_v2/checkpoint_best.pth \
--rayleigh 1e6 \
--prandtl 0.7 \
--eval_pseudo_batch_size 3000 \
--eval_xres 384 \
--eval_zres 128 \
--eval_tres 344 \
--Nens 1 \
--noiseTemp 0.0 \
--noiseVels 0.0 \
--ensembles True \
--eval_data Evaluation_rb2d_uniform_ra1e5_dt_0_1_t_50_60s_S_300.npz  \
--device cuda \
--output_folder out_test \
--seed 1 \
--n_samp_pts_per_crop 1024

