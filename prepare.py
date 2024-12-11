from dexgrasp.utils.general_utils import *

# create asset folder
os.makedirs(ASSET_DIR, exist_ok=True)
# copy hand assets
shutil.copytree(osp.join(BASE_DIR, 'dexgrasp/hand_assets/mjcf'), osp.join(ASSET_DIR, 'mjcf'))
shutil.copytree(osp.join(BASE_DIR, 'dexgrasp/hand_assets/urdf'), osp.join(ASSET_DIR, 'urdf'))
shutil.copytree(osp.join(BASE_DIR, 'dexgrasp/hand_assets/textures'), osp.join(ASSET_DIR, 'textures'))
# copy init assets
shutil.copyfile(osp.join(BASE_DIR, 'results/meshdatav3_pc_init.zip'), osp.join(ASSET_DIR, 'meshdatav3_pc_init.zip'))

# load configs 
config_state_based = load_yaml(osp.join(BASE_DIR, 'dexgrasp/cfg/train/universal_policy_state_based.yaml'))
config_vision_based = load_yaml(osp.join(BASE_DIR, 'dexgrasp/cfg/train/universal_policy_vision_based.yaml'))
# create log folder
log_dir_state_based = osp.join(LOG_DIR, config_state_based['Infos']['save_name'], 'results_distill/random', config_state_based['Distills']['save_name'], 'distill_0000_3199_seed0')
log_dir_vision_based = osp.join(LOG_DIR, config_vision_based['Infos']['save_name'], 'results_distill/random', config_vision_based['Distills']['save_name'], 'distill_0000_3199_seed0')
os.makedirs(log_dir_state_based, exist_ok=True)
os.makedirs(log_dir_vision_based, exist_ok=True)
# copy pre-trained checkpoints
shutil.copyfile(osp.join(BASE_DIR, 'results/state_based/model_best_isaacgym_3.pt'), osp.join(log_dir_state_based, 'model_best.pt'))
shutil.copyfile(osp.join(BASE_DIR, 'results/vision_based/model_best_isaacgym_3.pt'), osp.join(log_dir_vision_based, 'model_best.pt'))

