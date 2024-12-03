import os
import torch
import trimesh
import numpy as np
import pytorch3d.ops
import pytorch3d.structures
import pytorch_kinematics as pk

from utils.general_utils import *


class ShadowHandModel:
    def __init__(self, mjcf_path, mesh_path, simplify_mesh=False, n_surface_samples=1024, device='cuda:0'):
        """
        Create a Hand Model for a MJCF robot
       
        Parameters
        ----------
        mjcf_path: str, path to mjcf
        mesh_path: str, path to mesh
        simplify_mesh: False, simplify hand mesh
        n_surface_samples: int, number of surface samples
        device: str, torch.Device, device for torch tensors
        """
        # init device
        self.device = device
        # load articulation
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=self.device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # build mesh
        self.meshes, self.areas = {}, {}
        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices, link_faces, n_link_vertices = [], [], 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=self.device)
                    # load link mesh
                    if visual.geom_type == "box":
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, 'box.obj'), process=True)
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".stl"), process=True)
                        if visual.geom_param[1] is not None: scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=self.device)
                    elif visual.geom_type == None:
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".stl"), process=True)
                        if visual.geom_param[1] is not None: scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=self.device)
                    
                    # use open3d to simplify hand link_mesh
                    if simplify_mesh: link_mesh = simplify_trimesh(link_mesh, ratio=0.1)

                    # load vertices, faces
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=self.device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=self.device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    # append vertices, faces
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                # get link vertices
                link_vertices, link_faces = torch.cat(link_vertices, dim=0), torch.cat(link_faces, dim=0)
                self.meshes[link_name] = {'vertices': torch.cat([link_vertices, torch.ones((link_vertices.shape[0], 1), device=self.device)], dim=-1), 'faces': link_faces}
                # get link area
                self.areas[link_name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)
 
        # set joint limits
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []
        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(self.device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(self.device)
 
        # sample surface points
        total_area = sum(self.areas.values())
        num_samples = dict([(link_name, int(self.areas[link_name] / total_area * n_surface_samples)) for link_name in self.meshes])
        num_samples[list(num_samples.keys())[0]] += n_surface_samples - sum(num_samples.values())
        for link_name in self.meshes:
            if num_samples[link_name] == 0:
                self.meshes[link_name]['surface_points'] = torch.ones((1, 4), device=self.device)
                continue
            mesh = pytorch3d.structures.Meshes(self.meshes[link_name]['vertices'][..., :3].unsqueeze(0), self.meshes[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=self.device)
            self.meshes[link_name]['surface_points'] = torch.cat([surface_points, torch.ones((surface_points.shape[0], 1), device=self.device)], dim=-1)
        # # print hand model link info
        # for link_name in self.meshes: print('mesh vertices', link_name, self.meshes[link_name]['vertices'].shape, self.meshes[link_name]['faces'].shape, self.meshes[link_name]['surface_points'].shape, self.areas[link_name])
 
        # indexing
        self.link_name_to_link_index = dict(zip([link_name for link_name in self.meshes], range(len(self.meshes))))
       
        # parameters
        self.current_status = None
 
 
    def get_current_meshes(self, hand_pos, hand_rot, hand_pose):
        """
        Set translation, rotation, joint angles, and contact points of grasps
       
        Parameters
        ----------
        hand_pos: (B, 3), xyz
        hand_rot: (B, 4), quaternion
        hand_pose: (B, 22), joint_angle
        """
 
        # get batch_size
        batch_size = hand_pose.shape[0]
        # update current states
        current_meshes = {}
        self.current_status = self.chain.forward_kinematics(hand_pose)
        
        # get hand_matrix
        hand_matrix = torch.eye(4).repeat(batch_size, 1, 1).to(self.device)
        hand_matrix[:, :3, :3] = quaternion_to_matrix(hand_rot[:, [3, 0, 1, 2]])
        hand_matrix[:, :3, 3] = hand_pos
 
        # update current meshes
        for link_name, mesh_data in self.meshes.items():
            # init current link mesh
            current_meshes[link_name] = {}
            # get link transformation
            link_matrix = self.current_status[link_name].get_matrix()
            if link_matrix.shape[0] != batch_size: link_matrix = link_matrix.repeat(batch_size, 1, 1)
            # apply link transformation
            mesh_vertices = mesh_data['vertices'].repeat(batch_size, 1, 1)
            hand_vertices = torch.matmul(link_matrix, mesh_vertices.transpose(1, 2)).transpose(1, 2)[..., :3]
            # apply hand transformation
            temp_vertices = torch.cat([hand_vertices, torch.ones((hand_vertices.shape[0], hand_vertices.shape[1], 1), device=self.device)], dim=-1)
            current_meshes[link_name]['vertices'] = torch.matmul(hand_matrix, temp_vertices.transpose(1, 2)).transpose(1, 2)[..., :3]
            # append current_meshes faces
            current_meshes[link_name]['faces'] = mesh_data['faces'].repeat(batch_size, 1, 1)

        # combine current_meshes
        current_hand_vertices, current_hand_faces, num_current_vertices = [], [], 0
        for link_name, mesh_data in current_meshes.items():
            # append mesh vertices
            current_hand_vertices.append(mesh_data['vertices'])
            # append mesh faces with offset
            current_hand_faces.append(mesh_data['faces'] + num_current_vertices)
            # update offset
            num_current_vertices += mesh_data['vertices'].shape[1]
        # cat current hand vertices and faces
        current_hand_vertices = torch.cat(current_hand_vertices, dim=1)
        current_hand_faces = torch.cat(current_hand_faces, dim=1)
        return current_hand_vertices, current_hand_faces, current_meshes