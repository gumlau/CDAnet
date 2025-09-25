#!/bin/bash  

# set the name of datasets you would like to use for traning/validation
# train_dataset_name="rb2d_uniform_ra1e6_S_42.npz rb2d_uniform_ra1e6_S_52.npz"
# eval_dataset_name="rb2d_uniform_ra1e6_S_62.npz"

train_dataset_name="rb2d_uniform_ra1e6_S_10_chopped.npz rb2d_uniform_ra1e6_S_20_chopped.npz rb2d_uniform_ra1e6_S_30_chopped.npz rb2d_uniform_ra1e6_S_40_chopped.npz rb2d_uniform_ra1e6_S_50_chopped.npz rb2d_uniform_ra1e6_S_60_chopped.npz rb2d_uniform_ra1e6_S_70_chopped.npz rb2d_uniform_ra1e6_S_80_chopped.npz rb2d_uniform_ra1e6_S_90_chopped.npz"
eval_dataset_name="rb2d_uniform_ra1e6_S_100_chopped.npz "
DATA_FOLDER=/home/zhengf_lab/cse12210404/CDAnet/data

OUTPUT_FOLDER=testing_dataLoader_velOnly_v2

DEVICE=cuda
NUM_GPUS=4

# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=0.7
gamma=0.0125
use_continuity=true
log_dir_name="./log/test"
mkdir -p $log_dir_name

# CDA PARAMETERS
VEL_ONLY=true

MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

export OMP_NUM_THREADS=20

echo "[!] If you run into OOM error, try reducing batch_size_per_gpu or n_samp_pts_per_crop..."
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_addr $MY_MASTER_ADDR --master_port $MY_MASTER_PORT --use_env \
train_mine.py \
--epochs=100 \
--data_folder $DATA_FOLDER \
--alpha_pde=$gamma \
--train_data $train_dataset_name \
--eval_data $eval_dataset_name \
--rayleigh=$rayleigh \
--prandtl=$prandtl \
--nonlin=softplus \
--batch_size=6 \
--n_samp_pts_per_crop=1024 \
--use_continuity=$use_continuity \
--pseudo_epoch_size 3000 \
--loss_type "l1" \
--lr 0.01 \
--output_folder $OUTPUT_FOLDER \
--device $DEVICE \
--velocityOnly $VEL_ONLY

# please see the train.py for further tunable arguments during the training process
