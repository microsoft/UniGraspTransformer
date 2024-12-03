# bash train_offline_universal.sh 0 9 10 universal_policy_state_based.yaml train_set_results.yaml
# bash train_offline_universal.sh 0 139 140 universal_policy_state_based.yaml test_set_seen_cat_results.yaml
# bash train_offline_universal.sh 0 99 100 universal_policy_state_based.yaml test_set_unseen_cat_results.yaml
# Define Start, Finish Lines, Episode Num, Config_Dir, Object_File
Start=$1
Finish=$2
Episode=$3
Config_Dir=$4
Object_File=$5
Test_Epoch=$6
Model_Dir=$7
Device_Number=$(nvidia-smi --list-gpus | wc -l)

# No Test_Epoch Assigned
if [ -z "$Test_Epoch" ]; then
    Test_Epoch=0
fi

# No Model_Dir Assigned
if [ -z "$Model_Dir" ]; then
    # Default Model_Dir 
    Model_Dir="distill_$(printf "%04d" $Start)_$(printf "%04d" $Finish)"
    # Train MLP with Offline Trajectory
    python train_offline.py --start $Start --finish $Finish --config $Config_Dir --object $Object_File --device cuda:0
fi

# Define Hyper Params
Ntrain_its=10000
Ntest_envs=1000
Ntest_its=1

# Process Target Objects in Episode
Target_Episode_List=$(seq 1 $Episode)
Batch_Size=$(((Finish - Start + 1) / Episode))

# Test Target Objects with Episode
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
        
        echo "Running python test: $nline, episode: $nepisode, cuda:$cuda_id, config: $Config_Dir"
        # test distilled model for single line in object_scale_file, within Target_List
        python train.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:${cuda_id} \
        --num_envs $Ntest_envs --max_iterations $Ntrain_its --config $Config_Dir --headless --test --test_iteration $Ntest_its --test_epoch $Test_Epoch \
        --model_dir $Model_Dir --object_scale_file $Object_File --start_line $nline --end_line $((nline + 1))

        ) &
        
        ncount=$((ncount + 1))
    done
    wait
done
# plot results
python plot.py --subfolder results_distill/random --config $Config_Dir