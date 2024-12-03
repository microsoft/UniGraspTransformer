# bash train_online_dedicated.sh 0 9 5 dedicated_policy.yaml train_set_results.yaml
# Define Start, Finish Lines, Episode Num, Config_Dir, Object_File
Start=$1
Finish=$2
Episode=$3
Config_Dir=$4
Object_File=$5
Device_Number=$(nvidia-smi --list-gpus | wc -l)

# Define Hyper Params
Ntrain_envs=1000
Ntrain_its=10000
Ntest_envs=1000
Ntest_its=1

# Process Target Objects in Episode
Target_Episode_List=$(seq 1 $Episode)
Batch_Size=$(((Finish - Start + 1) / Episode))

# Train Target Objects with Episode
for nepisode in $Target_Episode_List
do
    # Update Target_Line_List at Current Episode
    Target_Line_List=$(seq $((Start + Batch_Size*(nepisode-1))) $((Start + Batch_Size*nepisode - 1)))
    
    ncount=0
    # Parallel Train Single Objects
    for nline in $Target_Line_List
    do
        (
        cuda_id=$((Device_Number - 1 - ncount % Device_Number))

        echo "Running python train: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # train single model for single line in object_scale_file, within Target_List
        python train_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs $Ntrain_envs --max_iterations $Ntrain_its --config $Config_Dir --headless \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1)) # --container

        echo "Running python test: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test single model for single line in object_scale_file, within Target_List
        python train_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs $Ntest_envs --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration $Ntest_its \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1)) # --save # --init # --container

        echo "Saving python test train: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test single model for single line in object_scale_file, within Target_List
        python train.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs 100 --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration 10 \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1)) --save --save_train # --init # --container

        ) &
        
        ncount=$((ncount + 1))
    done
    wait
done
# plot config result
python plot.py --config $Config_Dir --subfolder results_train