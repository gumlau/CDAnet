#!/bin/bash  

# OLD SET USABLE!
# train_dataset_name="rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_10.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_20.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_30.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_40.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_50.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_70.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_80.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_100.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_110.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_130.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_140.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_90.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_120.npz rb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_150.npz"
# eval_dataset_name="EVALrb2d_uniform_ra1e6_dt_0_1_t_0_50s_S_60.npz"

# NEW SET
train_dataset_name="rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_10.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_20.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_30.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_40.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_50.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_60.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_70.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_80.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_90.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_100.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_110.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_120.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_130.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_140.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_150.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_160.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_170.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_180.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_190.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_200.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_210.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_220.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_230.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_260.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_270.npz"
#  rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_280.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_290.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_300.npz"
#  rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_310.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_320.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_330.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_340.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_350.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_360.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_370.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_380.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_390.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_400.npz
eval_dataset_name="rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_240.npz rb2d_uniform_ra1e6_dt_0_05_t_35_50s_S_250.npz"

DATA_FOLDER=../data

# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=0.7
use_continuity=true

# Devices
DEVICE=cuda
NUM_GPUS=2
OUTPUT_FOLDER=paramSearch

# NUMERICAL PARAMETERS (DATA)
D_T=0.05
N_X=256.0
N_Z=256.0

# Crop Dimensions (Learning Related, How big is the data crop)
DT=16
NX=256
NZ=256

# CDA PARAMETERS
VEL_ONLY=false
factor_Spatial=8
factor_Temporal=8


# Training Parameters
EPOCHS=100
PSEUDO_EPOCH_SIZE=3000
#3000
BATCH_SIZE=15
NSPC=1024

MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

export OMP_NUM_THREADS=50


# GAMMA in 0.001 0.01 0.1 0.25 0.5 0.75 1.0 2.0
for GAMMA in 0.01 0.001 0.1
do
    for LR in 0.01 0.15 0.2 0.1 0.05 0.005
    # 0.01 0.001 0.0001 0.00001
    # for LR in 0.005 0.05 0.1
    do
        OUTPUT_FOLDER=inception_final_space${factor_Spatial}_time${factor_Temporal}/nt${DT}_nx${NX}_nz${NZ}_dXZ${factor_Spatial}_dT${factor_Temporal}_nppc${NSPC}_es${PSEUDO_EPOCH_SIZE}_bs${BATCH_SIZE}_lr${LR}_gamma${GAMMA}
        # paramSearch
        mkdir -p $OUTPUT_FOLDER

        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_addr $MY_MASTER_ADDR --master_port $MY_MASTER_PORT --use_env \
        train_mine.py \
        --epochs=$EPOCHS \
        --data_folder $DATA_FOLDER \
        --alpha_pde=$GAMMA \
        --train_data $train_dataset_name \
        --eval_data $eval_dataset_name \
        --rayleigh=$rayleigh \
        --prandtl=$prandtl \
        --nonlin=softplus \
        --batch_size=$BATCH_SIZE \
        --n_samp_pts_per_crop=$NSPC \
        --use_continuity=$use_continuity \
        --pseudo_epoch_size $PSEUDO_EPOCH_SIZE \
        --loss_type "l1" \
        --lr $LR \
        --output_folder $OUTPUT_FOLDER \
        --device $DEVICE \
        --velocityOnly $VEL_ONLY \
        --downsamp_xz $factor_Spatial \
        --downsamp_t $factor_Temporal \
        --nt $DT \
        --nx $NX \
        --nz $NZ 
    done
done


# echo "[!] If you run into OOM error, try reducing batch_size_per_gpu or n_samp_pts_per_crop..."
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_addr $MY_MASTER_ADDR --master_port $MY_MASTER_PORT --use_env \
# train_mine.py \
# --epochs=100 \
# --data_folder $DATA_FOLDER \
# --alpha_pde=$gamma \
# --train_data $train_dataset_name \
# --eval_data $eval_dataset_name \
# --rayleigh=$rayleigh \
# --prandtl=$prandtl \
# --nonlin=softplus \
# --batch_size=7 \
# --n_samp_pts_per_crop=1024 \
# --use_continuity=$use_continuity \
# --pseudo_epoch_size 3000 \
# --loss_type "l1" \
# --lr 0.01 \
# --output_folder $OUTPUT_FOLDER \
# --device $DEVICE \
# --velocityOnly $VEL_ONLY

# please see the train.py for further tunable arguments during the training process
