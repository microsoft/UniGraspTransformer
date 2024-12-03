import os
import time
import yaml
import json
import glob
import tqdm
import torch
import shutil
import psutil
import GPUtil
import pickle
import random
import trimesh
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
import os.path as osp

from PIL import Image
from typing import Dict, Any
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.transforms import euler_angles_to_matrix


# locate SupDexGrasp folder
LOG_DIR = "/data0/v-wenbowang/Desktop/Logs"
BASE_DIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
print('================ Run ================')
print('LOG_DIR:', LOG_DIR)
print('BASE_DIR', BASE_DIR)
print('================ Run ================')


# segmentation_id
SEGMENT_ID = {
    'back': [0, [255, 255, 255]],
    'table': [1, [0, 0, 0]],
    'hand': [2, [0, 255, 0]],
    'object': [3, [0, 0, 255]],
    'goal': [4, [125, 125, 125]],
    'other': [5, [125, 125, 125]],
    }
# segmentation_id_list
SEGMENT_ID_LIST = ['back', 'table', 'hand', 'object', 'goal', 'other']


# vision-based cameras
CAMERA_PARAMS = {
    'num': 6,
    'dist': 0.3,  # center distance
    'height': 0.15,  # center height

    'eye': np.array(
        [[ 0.3, 0.0, 0.3 ],  # monitor view
         [ 0.0, 0.0, 0.45 ],  # vertical view
         [ 0.3, 0.0, 0.15 ],  # horizontal views
         [ 0.0, 0.3, 0.15 ],
         [ -0.3, 0.0, 0.15 ],
         [ 0.0, -0.3, 0.15 ]]),

    'lookat': np.array([
        [ 0.0, 0.0, 0.2 ],
        [ 0.01, 0.0, 0.15 ],
        [ 0.0, 0.0, 0.15 ],
        [ 0.0, 0.0, 0.15 ],
        [ 0.0, 0.0, 0.15 ],
        [ 0.0, 0.0, 0.15 ],])
}

# # vision-based cameras
# CAMERA_PARAMS = {
#     'num': 6,

#     'eye': np.array(
#         [[ 0.3, 0.0, 0.3 ],  # global view
#          [ 0.0, 0.0, 0.55 ],
#          [ 0.5, 0.0, 0.05 ],
#          [ -0.5, 0.0, 0.05 ],
#          [ 0.0, 0.5, 0.05 ],
#          [ 0.0, -0.5, 0.05 ]]),

#     'lookat': np.array([
#         [ 0.0, 0.0, 0.2 ],
#         [ 0.01, 0.0, 0.05 ],
#         [ 0.0, 0.0, 0.05 ],
#         [ 0.0, 0.0, 0.05 ],
#         [ 0.0, 0.0, 0.05 ],
#         [ 0.0, 0.0, 0.05 ],])
# }

# scale string transfer
scale2str = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
str2scale = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}

# load yaml file
def load_yaml(path):
    with open(path, 'r') as file: 
        data = yaml.safe_load(file)
    return data

# save yaml file
def save_yaml(path, data):
    with open(path, 'w') as file: 
        yaml.safe_dump(data, file)


# load json file
def load_json(path):
    with open(path, 'r') as file: 
        data = json.load(file)
    return data

# save json file
def save_json(path, data):
    with open(path, 'w') as file: 
        json.dump(data, file)

# load data from pkl
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# save data to pkl
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))

# load a list of strings from txt
def load_list_strings(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return lines

# save a list of strings into txt
def save_list_strings(filename, data):
    with open(filename, "w") as file:
        for string in data:
            file.write(string + "\n")

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# show RGB image using plt, {q, s}
def show_image(img, name='image'):
    fig = plt.figure(figsize=(32, 18))
    fig.suptitle(name)
    plt.imshow(img)
    plt.show()

# draw points (Npoint, 2) on image
def draw_points(img, points, radius=1, color=[125, 125, 125]):
    for idx, point in enumerate(points):
        cv.circle(img, (int(point[1]), int(point[0])), radius=radius, color=color, thickness=-1)
    return img

# make video from image_fns
def make_video_from_image_fns(image_fns, output_fn, fps=30):
    # get image size
    frame = cv.imread(image_fns[0])
    height, width, _ = frame.shape

    # init video writer
    video_writer = cv.VideoWriter(output_fn, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # write each image
    for image_fn in image_fns: video_writer.write(cv.imread(image_fn))
    # release video writer
    video_writer.release()

# make video from image_fns
def make_video_from_images(images, output_fn, fps=30):
    # get image size
    height, width, _ = images[0].shape
    # init video writer
    video_writer = cv.VideoWriter(output_fn, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # write each image
    for image in images: video_writer.write(image[:, :, [2, 1, 0]])
    # release video writer
    video_writer.release()

# grid camera_images(n_cam, h, w) to size()
def grid_camera_images(images, size=[2, 3], border=False):
    if len(images) < size[0] * size[1]: size = [int(len(images) ** 0.5), int(len(images) ** 0.5)]
    grid_image = torch.cat([torch.cat([images[nw*size[0] + nh] for nh in range(size[0])], dim=0) for nw in range(size[1])], dim=1)
    if not border: return grid_image
    border_size = int(grid_image.shape[0] * 0.01)
    grid_image[:border_size, :] = 0
    grid_image[-border_size:, :] = 0
    grid_image[:, :border_size] = 0
    grid_image[:, -border_size:] = 0
    return grid_image

# load train_object_groups.yaml
def load_object_scale_group_yaml(file_path, group, nline=None):
    # init object_line_list, object_scale_list, object_scale_dict
    object_line_list, object_scale_list, object_scale_dict, result_dict = [], [], {}, {}
    # load object_scale_group
    object_scale_group = load_yaml(file_path)[group]
    # unpack object_scale_group
    for nobj in range(len(object_scale_group['object_line'])):
        # only load nline
        if nline is not None and nobj != nline: continue
        # append object_line and object_scale
        object_line = object_scale_group['object_line'][nobj]
        object_scale = object_scale_group['object_scale'][nobj]
        object_line_list.append(object_line)
        object_scale_list.append(object_scale)
        # append obj_name and scale_name
        object_scale_split = object_scale.split('/')
        obj_name, scale_name = '{}/{}'.format(object_scale_split[0], object_scale_split[1]), object_scale_split[2]
        if obj_name not in object_scale_dict: object_scale_dict[obj_name] = [float(scale_name)]
        else: object_scale_dict[obj_name].append(float(scale_name))
    return object_line_list, object_scale_list, object_scale_dict, result_dict

# load train_set_results.yaml
def load_object_scale_result_yaml(file_path, start=None, end=None, shuffle=False):
    # init object_line_list, object_scale_list, object_scale_dict
    success, index, count = 0, -1, 0
    object_line_list, object_scale_list, object_scale_dict, result_dict = [], [], {}, {}
    # get line_num and line_indices
    with open(file_path, 'r') as file:
        # get the number of lines
        line_num = sum(1 for line in file)
        if start is None or end is None: start, end = 0, line_num
        # get the index of lines
        if not shuffle: line_indices = [n for n in range(start, end)]
        else: line_indices = random.sample([n for n in range(line_num)], int(end-start))

    # load object_scale
    with open(file_path, 'r') as file:
        for line in file:
            index += 1
            if index not in line_indices: continue
            object_line_list.append(index)
            # 'core/bowl-c2882316451828fd7945873d861da519':[0.06],[0.981],
            line_items = line.split("'")
            obj_name = line_items[1]
            scale_name = line_items[2].split(',')[0][2:-1]
            result = float(line_items[2].split(',')[1][1:-1])
            # append item
            if obj_name not in object_scale_dict: object_scale_dict[obj_name] = [float(scale_name)]
            else: object_scale_dict[obj_name].append(float(scale_name))
            result_dict[obj_name+'/'+scale_name] = result
            object_scale_list.append(obj_name+'/'+scale_name)
            success += result
            count += 1

    print('{}, {}, success_mean: {}'.format(file_path.split('/')[-1], count, success / count))
    return object_line_list, object_scale_list, object_scale_dict, result_dict

# print cpu and gpu usage
def print_cpu_gpu_usage(print_cpu=False, print_gpu=False, gpu_id=0):
    # init usage
    usage = []
    # print cpu usage
    if print_cpu:
        # get maximum cpu memory usage
        virtual_memory = psutil.virtual_memory()
        total_memory = virtual_memory.total / (1024 ** 2)
        percent = psutil.virtual_memory().percent
        used = psutil.virtual_memory().used / (1024 ** 2)
        usage.append("CPU Memory usage: {}% {:.2f} / {:.2f} MB".format(percent, used, total_memory))
        print(usage[-1])
    # print gpu usage
    if print_gpu:
        # get current gpu memory usage
        gpu = GPUtil.getGPUs()[gpu_id]
        usage.append("GPU Memory usage: {} {:.2f}/{:.2f} MB".format(gpu.id, gpu.memoryUsed, gpu.memoryTotal))
        print(usage[-1])
    return usage


# compute encoding vector (nenv, dimension) for time (nenv)
def compute_time_encoding(time, dimension):
    # Create a tensor for dimension indices: [0, 1, 2, ..., dimension-1]
    div_term = torch.arange(0, dimension, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / dimension)
    div_term = torch.exp(div_term).unsqueeze(0).to(time.device)  # Shape: (1, dimension/2)
    # Apply sin to even indices in the array; 2i
    encoding = torch.zeros(time.shape[0], dimension).to(time.device)
    encoding[:, 0::2] = torch.sin(time.unsqueeze(1) * div_term)
    # Apply cos to odd indices in the array; 2i+1
    encoding[:, 1::2] = torch.cos(time.unsqueeze(1) * div_term)
    return encoding

# compute trajectory valids from object_pos(ntraj, nstep, 3)
def compute_trajectory_valids(object_pos, target_pos=np.array([0, 0, 0.9]), max_goal_dist=0.05, max_valid_length=60):
    # get goal distances
    successes = np.linalg.norm(object_pos - target_pos, axis=-1) <= max_goal_dist
    # get successes indices
    successes_indices = np.argmax(successes, axis=1)
    reversed_successes_indices = successes.shape[1] - 1 - np.argmax(np.flip(successes, axis=1), axis=1)
    successes_indices[np.all(successes == False, axis=1)] = -1
    reversed_successes_indices[np.all(successes == False, axis=1)] = -1

    # get valids with max_valid_length
    valids = np.zeros((object_pos.shape[0], object_pos.shape[1], 1))
    for ntraj in range(valids.shape[0]):
        if successes_indices[ntraj] == -1: continue
        index = min(reversed_successes_indices[ntraj], successes_indices[ntraj] + max_valid_length)
        valids[ntraj, :index] = 1.
    return valids


# simplify trimesh vertices
def simplify_trimesh(mesh, ratio=0.1, min_faces=None):
    # # simplify trimesh
    # mesh = mesh.simplify_quadric_decimation(1)
    # init open3d mesh
    temp_mesh = o3d.geometry.TriangleMesh()
    temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    # use open3d to simplify mesh
    num_faces = int(len(temp_mesh.triangles)*ratio)
    if min_faces is not None: num_faces = max(min_faces, num_faces)
    temp_mesh = temp_mesh.simplify_quadric_decimation(target_number_of_triangles=num_faces)
    # return trimesh
    return trimesh.Trimesh(vertices=np.asarray(temp_mesh.vertices), faces=np.asarray(temp_mesh.triangles), process=True)

# sample points using pointnet2_utils
def sample_points(points, sample_num, sample_method, device):
    from pointnet2_ops import pointnet2_utils
    # random method
    if sample_method == 'random':
        indices = torch.randint(0, points.shape[1], (sample_num,))
        sampled_points = points[:, indices, :]
    # furthest batch
    elif sample_method == "furthest_batch":
        idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous(), sample_num).long()
        idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
        sampled_points = torch.gather(points, dim=1, index=idx)
    # furthest
    elif sample_method == 'furthest':
        eff_points = points[points[:, 2] > 0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num:
            eff_points = points[:, 0:3].contiguous()
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points_xyz.reshape(1, *eff_points_xyz.shape), sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
    else:
        assert False
    return sampled_points

# sample label points within points (Nbatch, Npoint, 4)
def sample_label_points(points, label, number):
    # get nbatch, npoint, ndata
    nbatch, npoint, ndata = points.shape
    # get label_flags
    label_flags = points[..., -1] == label
    # init label_points
    label_points, appears = [], torch.ones(nbatch).to(points.device)
    for n in range(nbatch):
        num_sample = torch.sum(label_flags[n])
        # zeros points with label
        if num_sample == 0: samples, appears[n] = torch.zeros((number, ndata), dtype=points.dtype).to(points.device), 0
        # label points with label
        else: samples = points[n][label_flags[n]][torch.randint(0, num_sample, (number, ))]
        # append label_points
        label_points.append(samples)
    return torch.stack(label_points), appears

# check object appears, update disappear datas
def check_object_valid_appears(valids, datas):
    # disappears: valid step and disappear object_pc
    disappears = ((valids == 1) * 1) * ((datas['appears'] == 0) * 1)
    if np.sum(disappears) > 0:
        # process all trajectory
        for ntraj in range(disappears.shape[0]):
            if np.sum(disappears[ntraj]) == 0: continue
            # locate disappear object steps
            steps = np.sort(np.nonzero(disappears[ntraj])[0])
            # disappear first frame: leave appears to be zero
            if steps[0] == 0: continue
            # disappear middle frames: update with previous frame
            for _, step in enumerate(steps):
                for key, value in datas.items(): datas[key][ntraj, step] = datas[key][ntraj, step-1]
    return datas

# project points(Npoint, 3) on image(1024, 1024, 3)
def project_points_on_images(points, image_size, view_mat, proj_matrix):
    # project points to image space
    points = torch.cat([points, torch.ones(points.shape[0], 1).to(points.device)], dim=1)
    points = points @ view_mat
    points = points @ proj_matrix
    ndc_space_points = points / points[..., 3].unsqueeze(-1)
    # normalize image_coords 
    image_coords = torch.zeros(points.shape[0], 2)
    image_coords[:, 0] = (ndc_space_points[:, 0] + 1) * 0.5 * image_size[0]  # x-coordinates (scale and shift)
    image_coords[:, 1] = (1 - ndc_space_points[:, 1]) * 0.5 * image_size[1]  # y-coordinates (invert y-axis, scale and shift)
    # clip to ensure all coordinates fall within the image bounds
    image_coords[:, 0] = torch.clamp(image_coords[:, 0], 0, image_size[0] - 1)
    image_coords[:, 1] = torch.clamp(image_coords[:, 1], 0, image_size[1] - 1)
    return image_coords[:, [1, 0]]

# convert depth image to image points
def depth_image_to_point_cloud_GPU_batch(
    # camera_depth_tensor_batch, camera_rgb_tensor_batch, camera_seg_tensor_batch, 
    camera_depth_tensor_batch, camera_seg_tensor_batch, 
    camera_view_matrix_inv_batch, camera_proj_matrix_batch, u, v, 
    width: float, height: float, depth_bar: float, device: torch.device,
    # z_p_bar: float = 3.0,
    # z_n_bar: float = 0.3,
):
    # get batch_size
    batch_num = camera_depth_tensor_batch.shape[0]
    # move depth, rgb, and seg tensor to device
    depth_buffer_batch = camera_depth_tensor_batch
    # rgb_buffer_batch = camera_rgb_tensor_batch / 255.0
    seg_buffer_batch = camera_seg_tensor_batch

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch
    # Get the camera projection matrix and get the necessary scaling coefficients for deprojection
    proj_batch = camera_proj_matrix_batch
    fu_batch = 2 / proj_batch[:, 0, 0]
    fv_batch = 2 / proj_batch[:, 1, 1]
    centerU = width / 2
    centerV = height / 2
    # Unpack depth tensor 
    Z_batch = depth_buffer_batch
    Z_batch = torch.nan_to_num(Z_batch, posinf=1e10, neginf=-1e10)
    X_batch = -(u.view(1, u.shape[-2], u.shape[-1]) - centerU) / width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1]) - centerV) / height * Z_batch * fv_batch.view(-1, 1, 1)
    # # Unpack rgb tensor
    # R_batch = rgb_buffer_batch[..., 0].view(batch_num, 1, -1)
    # G_batch = rgb_buffer_batch[..., 1].view(batch_num, 1, -1)
    # B_batch = rgb_buffer_batch[..., 2].view(batch_num, 1, -1)
    # Unpack seg tensor
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)
    
    # Locate valid depth tensor
    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar
    
    # Pack position_batch (batch_num, 8, N)
    Z_batch = Z_batch.view(batch_num, 1, -1)
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)
    # position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, S_batch), dim=1)
    # Project depth pixel position from image space to world space (b, N, :4)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[..., 0:4] = position_batch[..., 0:4] @ vinv_batch
    # Final points: X, Y, Z, R, G, B, S
    # points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
    points_batch = position_batch[..., [0, 1, 2, 4]]
    # Final valid flag
    valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch
    return points_batch, valid_batch


@torch.jit.script
def batch_quat_apply(a, b):
    # unsqueeze a(Nenv, 1, 4)
    shape = b.shape
    a = a.unsqueeze(1)
    # extract the xyz component of quaternion a
    xyz = a[:, :, :3]
    # compute the cross product t
    t = torch.cross(xyz, b, dim=-1) * 2
    # compute the final result and reshape it to the original shape
    return (b + a[:, :, 3:] * t + torch.cross(xyz, t, dim=-1)).view(shape)

@torch.jit.script
# compute sided distance from sources(Nenv, Ns, 3) to targets(Nenv, Nt, 3)
def batch_sided_distance(sources, targets):
    # pairwise_distances: (Nenv, Ns, Nt)
    pairwise_distances = torch.cdist(sources, targets)
    # find the minimum distances
    distances, _ = torch.min(pairwise_distances, dim=-1)
    return distances

# compute the pca axes from inputs(Nenv, Npoint, 3)
def batch_decompose_pcas(inputs):
    # center inputs
    center_inputs = (inputs - torch.mean(inputs, dim=1).unsqueeze(1)).cpu().numpy()
    # extract object pca axes
    pca_axes, pca_agent = [], PCA(n_components=3)
    for n in range(center_inputs.shape[0]):
        pca_agent.fit(center_inputs[n])
        pca_axes.append(pca_agent.components_)
    # return pca_axes(Nenv, 3, 3)
    return np.stack(pca_axes, axis=0)

# set worker seeds
def worker_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# init worker seed
def worker_init(worker_id, main_seed):
    seed = worker_id + main_seed
    worker_seed(seed)

# set global seed
def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed
