GPU_ID=0

###################  ScanNet dataset  ######################
DATASET='scannet'
SPLIT=1    #  options: {0, 1}
DATA_PATH='/media/sda/point_cloud_datasets/ScanNet/blocks_bs1_s1'
SAVE_PATH='./log_scannet_SegPN/'


NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
WAY_SAMP_RATIO='[0.05, 0.05]'
WAY_SAMP_NUM='[100, 100]'

N_WAY=3    #  options: {2, 3}
K_SHOT=5    #  options: {1, 5}
N_QUESIES=1
N_TEST_EPISODES=100

args=(--model 'segpn_eval' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --way_pcratio "$WAY_SAMP_RATIO" --way_pcnum "$WAY_SAMP_NUM"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --batch_size 1 --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
