#!/bin/bash  

# set the name of datasets you would like to use for traning/validation
#train_dataset_name="rb2d_uniform_ra1e6_S_42.npz rb2d_uniform_ra1e6_S_52.npz"
#eval_dataset_name="rb2d_uniform_ra1e6_S_62.npz"

train_dataset_name="rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_10.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_100.npz"
eval_dataset_name="EVAL_rb2d_uniform_ra1e6_dt_0_1_t_30_45s_S_270.npz"
DATA_FOLDER=../data
# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1
gamma=0.0125
use_continuity=true
log_dir_name="./log/Exp1"
mkdir -p $log_dir_name

echo "[!] If you run into OOM error, try reducing batch_size_per_gpu or n_samp_pts_per_crop..."
CUDA_VISIBLE_DEVICES=0,1 python3 train_mine.py --epochs=100 --data_folder $DATA_FOLDER --log_dir=$log_dir_name --alpha_pde=$gamma --train_data $train_dataset_name --eval_data $eval_dataset_name --rayleigh=$rayleigh --prandtl=$prandtl --nonlin=swish --batch_size_per_gpu=4 --n_samp_pts_per_crop=1024 --use_continuity=$use_continuity \
--pseudo_batch_size 256 --reg_loss_type "l2" --lr 0.005

# please see the train.py for further tunable arguments during the training process
