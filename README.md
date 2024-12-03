# UniGraspTransformer
UniGraspTransformer Code Repository

# Folder Structure:
```
PROJECT_BASE
    └── Logs
        └── Results
            └── results_train
            └── results_distill
            └── results_trajectory
    └── Assets
        └── datasetv4.1_posedata.npy
        └── meshdatav3_pc_feat
        └── meshdatav3_scaled
        └── meshdatav3_init
        └── textures
        └── mjcf
        └── urdf
    └── isaacgym3
    └── isaacgym4
    └── UniGraspTransformer
        └── results
        └── dexgrasp
```

# Env Install:
Create conda env:
```
conda create -n dexgrasp python=3.8
conda activate dexgrasp
```

Install isaacgym3
```
cd PROJECT_BASE/isaacgym3/isaacgym3/python
pip install -e .
```

Install UniGraspTransformer
```
cd PROJECT_BASE/UniGraspTransformer
pip install -e .
```

Install pytorch3d and pytorch_kinematics
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
cd PROJECT_BASE/UniGraspTransformer/pytorch_kinematics
pip install -e .
```

# Step1: Train and Test Dedicated Policy:
```
cd PROJECT_BASE/UniGraspTransformer/dexgrasp/
```

Train&Test dedicated policy for single $nline=0 object in $Object_File=train_set_results.yaml:
```
python train.py --task=StateBasedGrasp --algo=ppo --seed=0 --rl_device cuda:0 \
    --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --headless \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
```
```
python train.py --task=StateBasedGrasp --algo=ppo --seed=0 --rl_device cuda:0 \
    --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --headless --render_hyper_view --test --test_iteration 1 \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
```

# Step2: Generate Trajectory Dataset:
Generate trajectories for single $nline=0 object in $Object_File=train_set_results.yaml:
```
python train.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 \
    --num_envs 100 --max_iterations 10000 --config dedicated_policy.yaml --headless --test --test_iteration 10 \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1 --save --save_train
```

# Step3: Train and Test Universal Policies:
Repeat step1 and step2 for $nline objects, like from 0 to 9, and train universal policies:
'''
python train_offline.py --start 0 --finish 9 --config universal_policy_state_based.yaml --object train_set_results.yaml --device cuda:0
python train_offline.py --start 0 --finish 9 --config universal_policy_vision_based.yaml --object train_set_results.yaml --device cuda:0
'''
Test state-based universal policy on $nline=0 object.
'''
python train.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 \
--num_envs 100 --max_iterations 10000 --config universal_policy_state_based.yaml --headless --test --test_iteration 1 \
--model_dir distill_0000_0009 --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
'''
Test vision-based universal policy on $nline=0 object.
'''
python train.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 \
--num_envs 100 --max_iterations 10000 --config universal_policy_vision_based.yaml --headless --test --test_iteration 1 \
--model_dir distill_0000_0009 --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
'''
