# bash train_pointnet.sh Object 0 9 5
# python train_pointnet.py --mode Object --start 0 --finish 9 --batch_size 5

Mode=$1
Start=$2
Finish=$3
Batch_Size=$4

# Train Server with Offline Trajectory: 8 GPUs
accelerate launch \
    --multi_gpu --num_machines 1 --num_processes 8 \
    --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    train_pointnet.py --mode $Mode --start $Start --finish $Finish --batch_size $Batch_Size

# # Train Local with Offline Trajectory: 4 GPUs
# accelerate launch \
#     --multi_gpu --num_machines 1 --num_processes 4 \
#     --machine_rank 0 --main_process_ip 127.0.0.1 --main_process_port 2097 \
#     train_pointnet.py --mode $Mode --start $Start --finish $Finish --batch_size $Batch_Size