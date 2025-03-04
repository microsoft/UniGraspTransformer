# # Train, Test, Generate Trajectory from scratch:
# bash run_online_parallel.sh 0 9 10 dedicated_policy.yaml train_set_results.yaml

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

        echo "Train dedicated policy: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # train single model for single line in object_scale_file, within Target_List
        python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs $Ntrain_envs --max_iterations $Ntrain_its --config $Config_Dir --headless \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1))

        echo "Test dedicated policy: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test single model for single line in object_scale_file, within Target_List
        python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs $Ntest_envs --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration $Ntest_its \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1))

        echo "Render dedicated policy: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test single model for single line in object_scale_file, within Target_List
        python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs 9 --max_iterations $Ntrain_its --config $Config_Dir --headless --render_hyper_view --test --test_iteration 1 \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1))

        echo "Generate trajectory for State-Based (results_trajectory_train) (Without Render: Fast): $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test single model for single line in object_scale_file, within Target_List
        python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs 100 --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration 10 \
        --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1)) --save --save_train # --save_render

        # echo "Generate trajectory for Vision-Based (results_trajectory_render) (With Render: Slow): $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # # test single model for single line in object_scale_file, within Target_List
        # python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:${cuda_id} \
        # --num_envs 100 --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration 10 \
        # --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1)) --save --save_train --save_render

        ) &
        
        ncount=$((ncount + 1))
    done
    wait
done
# plot config result
python plot.py --config $Config_Dir --subfolder results_train