from utils.general_utils import *
from utils.hand_model import ShadowHandModel

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    HardPhongShader,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    TexturesVertex,
    TexturesUV,
)

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform, 
    FoVPerspectiveCameras, 
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor
)


class PytorchBatchRenderer:
    """
    Pytorch3d Multi-View Image Renderer.
    """

    def __init__(self, num_view=6, img_size=256, center=None, max_x=1.0, min_x=-1.0, max_y=1.0, min_y=-1.0, dtype=torch.float32, device='cuda:0'):

        # init dtype, device
        self.dtype = dtype
        self.device = device

        # init num_view, img_size
        self.num_view = num_view
        self.img_size = img_size
        # # init range for FoVOrthographicCameras
        # self.range_ratio = 1
        # self.max_x, self.min_x = max_x * self.range_ratio, min_x * self.range_ratio
        # self.max_y, self.min_y = max_y * self.range_ratio, min_y * self.range_ratio

        # get camera dist and height, apply center offset
        dist, height = CAMERA_PARAMS['dist'], CAMERA_PARAMS['height']
        # init monitor view with dist and height
        R_m, T_m = look_at_view_transform(dist=(0.1**2+0.3**2)**0.5, elev=np.degrees(np.arctan(1/3)), azim=0, up=((0, 1, 0),), at=((0, height+center[-1]+0.05, 0),))
        # init vertical view with dist and height
        R_v, T_v = look_at_view_transform(dist=dist, elev=90, azim=0, up=((0, 1, 0),), at=((0, height+center[-1], 0),))
        # init horizontal views with num_view, dist and height
        azim = torch.linspace(0, 360, 5)[:4]
        R_h, T_h = look_at_view_transform(dist=dist, elev=0, azim=azim, up=((0, 1, 0),), at=((0, height + center[-1], 0),))
        # init final R and T
        R, T = torch.cat([R_m, R_v, R_h]), torch.cat([T_m, T_v, T_h])

        # init PointLights
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 1.0]], ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
        # init RasterizerSettings
        self.mesh_raster_settings = RasterizationSettings(image_size=self.img_size, faces_per_pixel=1, bin_size=0, blur_radius=0)  # faces_per_pixel=10
        self.point_raster_settings = PointsRasterizationSettings(image_size=self.img_size, radius=0.004, points_per_pixel=1)

        # init mesh_renderer and mesh_rasterizer
        self.mesh_renderer, self.mesh_rasterizer, self.point_renderer = [], [], []
        self.camera_view_mat, self.camera_vinv_mat, self.camera_proj_matrix = [], [], []
        for n_view in range(self.num_view):
            # init perspective camera
            camera = FoVPerspectiveCameras(fov=90, znear=0.001, zfar=1000.0, R=R[n_view].unsqueeze(0), T=T[n_view].unsqueeze(0), device=self.device)
            # camera = FoVOrthographicCameras(max_x=self.max_x, min_x=self.min_x, max_y=self.max_y, min_y=self.min_y, R=R[n_view].unsqueeze(0), T=T[n_view].unsqueeze(0), device=self.device)
            # init mesh_rasterizer 
            mesh_rasterizer = MeshRasterizer(cameras=camera, raster_settings=self.mesh_raster_settings)
            # init mesh_renderer
            shader = HardPhongShader(cameras=camera, lights=self.lights, device=self.device)
            mesh_renderer = MeshRenderer(rasterizer=mesh_rasterizer, shader=shader)
            # append mesh_renderer and mesh_rasterizer
            self.mesh_renderer.append(mesh_renderer)
            self.mesh_rasterizer.append(mesh_rasterizer)

            # append point_renderer
            point_renderer = PointsRenderer(rasterizer=PointsRasterizer(cameras=camera, raster_settings=self.point_raster_settings), compositor=AlphaCompositor())
            self.point_renderer.append(point_renderer)

            # get camera view_mat and proj_matrix
            self.camera_view_mat.append(camera.get_world_to_view_transform().get_matrix())
            self.camera_proj_matrix.append(camera.get_projection_transform().get_matrix())

        # cat camera_vinv_mat and camera_proj_matrix
        self.camera_view_mat = torch.cat(self.camera_view_mat)
        self.camera_vinv_mat = torch.inverse(self.camera_view_mat)
        self.camera_proj_matrix = torch.cat(self.camera_proj_matrix)
        
        # load goal_table_image as background
        self.goal_table_image = load_image(osp.join(BASE_DIR, "dexgrasp/hand_assets/goal_table.png"))
        self.goal_table_image = cv.resize(self.goal_table_image, (self.img_size, self.img_size))


    # pack pytorch meshes with verts(B, Nverts, 3)
    def pack_rendered_meshes(self, verts, faces, colors=None):
        if colors is None: return Meshes(verts, faces)
        else: return Meshes(verts, faces, textures=TexturesVertex(verts_features=colors))

    @torch.no_grad()
    # render mesh to multi_view images(Nbatch, Nview, H, W, RGBMD)
    def render_mesh_images(self, verts, faces, colors=None):
        # init rendered_images
        rendered_images = []
        # pack rendered_meshes, verts(Nbatch, Nverts, 3)
        rendered_meshes = self.pack_rendered_meshes(verts, faces, colors)
        # render images for num_view
        for n_view in range(self.num_view):
            # render color image
            color_image = self.mesh_renderer[n_view](rendered_meshes.extend(1))
            # render depth image
            rasterizer = self.mesh_rasterizer[n_view](rendered_meshes.extend(1))
            depth_image = rasterizer.zbuf[..., 0].unsqueeze(-1)            
            depth_image[depth_image!=-1] *= -1
            # append rendered_images
            rendered_images.append(torch.cat([color_image, depth_image], dim=-1))
        return torch.stack(rendered_images, dim=1)

    @torch.no_grad()
    # render mesh to multi_view depth images(Nbatch, Nview, H, W, DM)
    def render_mesh_depth_images(self, verts, faces):
        # init rendered_images
        rendered_images = []
        # pack rendered_meshes, verts(Nbatch, Nverts, 3)
        rendered_meshes = self.pack_rendered_meshes(verts, faces, colors=None)
        # render images for num_view
        for n_view in range(self.num_view):
            # render depth image
            rasterizer = self.mesh_rasterizer[n_view](rendered_meshes.extend(1))
            depth_image = rasterizer.zbuf[..., 0].unsqueeze(-1)            
            depth_image[depth_image!=-1] *= -1
            # render mask image: back-0, other-1
            mask_image = (depth_image!=-1) * 1
            # append rendered_images
            rendered_images.append(torch.cat([depth_image, mask_image], dim=-1))
        return torch.stack(rendered_images, dim=1)
    
    @torch.no_grad()
    # render points to multi_view images(Nbatch, Nview, H, W, RGBM)
    def render_point_images(self, points, colors):
        # init rendered_images
        rendered_images = []
        # init point_cloud
        point_cloud = Pointclouds(points=points, features=colors)
        # render images for num_view
        for n_view in range(self.num_view):
            # render color_image
            color_image = self.point_renderer[n_view](point_cloud.extend(1))
            rendered_images.append(color_image)
        return torch.stack(rendered_images, dim=1)


class TrajectoryRenderer:
    """
    Hand Object Trajectory Renderer.
    """

    def __init__(self, num_view=6, img_size=256, center=np.array([0., 0., 0.6]), dtype=torch.float32, device='cuda:0'):

        # init dtype, device
        self.dtype = dtype
        self.device = device
        # init num_view, img_size
        self.num_view = num_view
        self.img_size = img_size
        # init center
        self.center = center

        # init shadow_hand_model
        self.shadow_hand_model = ShadowHandModel('./hand_assets/shadow_hand_render.xml', './hand_assets/open_ai_assets/stls/hand', simplify_mesh=True, device=self.device)

        # init pytorch_renderer
        self.pytorch_renderer = PytorchBatchRenderer(num_view=self.num_view, img_size=self.img_size, center=self.center, device=self.device)
        # load pytorch_renderer view_matrix, convert to isaacgym axis
        self.pytorch_renderer_view_matrix = self.pytorch_renderer.camera_view_mat
        self.pytorch_renderer_view_matrix[:, :, [0, 2]] *= -1
        self.pytorch_renderer_view_matrix = self.pytorch_renderer_view_matrix[:, [2, 0, 1, 3], :]
        # load pytorch_renderer proj_matrix, convert to isaacgym axis
        self.pytorch_renderer_proj_matrix = self.pytorch_renderer.camera_proj_matrix
        self.pytorch_renderer_proj_matrix[:, [2, 3], :] *= -1
        # load pytorch_renderer proj_matrix
        self.pytorch_renderer_vinv_matrix = torch.inverse(self.pytorch_renderer_view_matrix)
        
        # create camera uv
        self.camera_v2, self.camera_u2 = torch.meshgrid(torch.arange(0, self.img_size), torch.arange(0, self.img_size), indexing='ij')
        self.camera_v2, self.camera_u2 = self.camera_v2.to(self.device), self.camera_u2.to(self.device)
        # set sample parameters
        self.num_pc_presample, self.num_pc_downsample = 65536, 1024

        # init table_mesh
        self.table_mesh = trimesh.creation.box(extents=(1., 1., self.center[-1]))
        self.table_mesh.vertices += 0.5 * self.center
        # create table_mesh vertices, faces
        self.table_mesh_vertices = torch.tensor(self.table_mesh.vertices, dtype=torch.float).unsqueeze(0).to(self.device)
        self.table_mesh_faces = torch.tensor(self.table_mesh.faces, dtype=torch.long).unsqueeze(0).to(self.device)
    

    # load and simplify object trimesh
    def load_object_mesh(self, object_mesh_file):
        # load and simplify object mesh
        object_mesh = simplify_trimesh(trimesh.load(object_mesh_file), ratio=0.1, min_faces=500)
        # return object_mesh vertices, faces
        object_mesh_vertices = torch.tensor(object_mesh.vertices, dtype=torch.float).unsqueeze(0).to(self.device)
        object_mesh_faces = torch.tensor(object_mesh.faces, dtype=torch.long).unsqueeze(0).to(self.device)
        return object_mesh_vertices, object_mesh_faces


    # render hand_object_states(Nbacth, [3, 4, 22, 3, 4]) into multi_view images
    def render_hand_object_states(self, hand_object_states, object_mesh_vertices, object_mesh_faces, render_images=False, sample_points=False):
        # return without rendering
        if not render_images: return None, None

        # get batch_size
        batch_size, _ = hand_object_states.shape
        # unpack hand_object_states: [3, 4, 22, 3, 4]
        hand_pos, hand_rot, hand_pose = hand_object_states[:, :3], hand_object_states[:, 3:3+4], hand_object_states[:, 3+4:3+4+22]
        object_pos, object_rot = hand_object_states[:, 3+4+22:3+4+22+3], hand_object_states[:, 3+4+22+3:]

        # get current shadow_hand vertices and faces
        self.shadow_hand_vertices, self.shadow_hand_faces, _ = self.shadow_hand_model.get_current_meshes(hand_pos, hand_rot, hand_pose)
        self.shadow_hand_colors = torch.tensor(SEGMENT_ID['hand'][1]).repeat(self.shadow_hand_vertices.shape[0], self.shadow_hand_vertices.shape[1], 1).to(self.device) / 255.

        # get current object vertices and faces
        self.object_vertices = object_mesh_vertices if object_mesh_vertices.shape[0] == batch_size else object_mesh_vertices.repeat(batch_size, 1, 1)
        self.object_vertices = batch_quat_apply(object_rot, self.object_vertices) + object_pos.unsqueeze(1)
        self.object_faces = object_mesh_faces if object_mesh_faces.shape[0] == batch_size else object_mesh_faces.repeat(batch_size, 1, 1)
        self.object_colors = torch.tensor(SEGMENT_ID['object'][1]).repeat(self.object_vertices.shape[0], self.object_vertices.shape[1], 1).to(self.device) / 255.

        # get current table vertices and colors
        self.table_vertices = self.table_mesh_vertices.repeat(batch_size, 1, 1)
        self.table_faces = self.table_mesh_faces.repeat(batch_size, 1, 1)
        self.table_colors = torch.tensor(SEGMENT_ID['table'][1]).repeat(batch_size, self.table_vertices.shape[1], 1).to(self.device) / 255.

        # combine shadow_hand, object, table meshes
        self.rendered_mesh_vertices = torch.cat([self.shadow_hand_vertices, self.object_vertices, self.table_vertices], dim=1)
        self.rendered_mesh_faces = torch.cat([self.shadow_hand_faces, self.object_faces+self.shadow_hand_vertices.shape[1], self.table_faces+self.shadow_hand_vertices.shape[1]+self.object_vertices.shape[1]], dim=1)
        self.rendered_mesh_colors = torch.cat([self.shadow_hand_colors, self.object_colors, self.table_colors], dim=1)

        # render images (Nenv, Nview, H, W, RGBMD)
        rendered_images = self.pytorch_renderer.render_mesh_images(self.rendered_mesh_vertices[:, :, [1, 2, 0]], self.rendered_mesh_faces, self.rendered_mesh_colors)
        # rendered labels (Nenv, Nview, H, W)
        segmentation_labels = torch.stack([torch.tensor(SEGMENT_ID[label][1]) for label in SEGMENT_ID_LIST]).to(self.device) / 255.
        rendered_labels = torch.argmin(torch.norm(rendered_images[..., :3].unsqueeze(-2).repeat(1, 1, 1, 1, segmentation_labels.shape[0], 1) - segmentation_labels.reshape(1, 1, 1, 1, segmentation_labels.shape[0], segmentation_labels.shape[1]), dim=-1), dim=-1)
        # get final rendered_images (Nenv, Nview, H, W, RGBMDS)
        rendered_images = torch.cat([rendered_images, rendered_labels.unsqueeze(-1)], dim=-1)

        # return without sampling
        if not sample_points: return rendered_images, None
        # repeat pytorch_renderer params with batch_size
        pytorch_renderer_proj_matrix = self.pytorch_renderer_proj_matrix.repeat(batch_size, 1, 1, 1)
        pytorch_renderer_vinv_matrix = self.pytorch_renderer_vinv_matrix.repeat(batch_size, 1, 1, 1)
        # render point_clouds (Nenv, Npoint, XYZS)
        rendered_points, others = self.render_camera_point_clouds(rendered_images[..., -2], rendered_images[..., -1], # self.vinv_mat, self.proj_matrix)
                                                                  pytorch_renderer_vinv_matrix, pytorch_renderer_proj_matrix, render_scene_only=True)

        return rendered_images, rendered_points

    # render scene, hand, object point clouds
    def render_camera_point_clouds(self, depth_tensor, seg_tensor, vinv_mat, proj_matrix, render_scene_only=True):
        # init point and valid list
        batch_size, point_list, valid_list = depth_tensor.shape[0], [], []
        # get pixel point from depth, rgb, and seg images
        for i in range(1, depth_tensor.shape[1]):
            # (num_envs, num_pts, 4) (num_envs, num_pts)
            point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], seg_tensor[:, i],
                                                                vinv_mat[:, i], proj_matrix[:, i], self.camera_u2, self.camera_v2, 
                                                                self.img_size, self.img_size, 1, self.device)
            point_list.append(point)
            valid_list.append(valid)

        # shift points (num_envs, 256*256 * num_cameras, 4)
        points = torch.cat(point_list, dim=1)
        # points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3)
        # get final valid mask
        depth_mask = torch.cat(valid_list, dim=1)
        s_mask = ((points[:, :, -1] == SEGMENT_ID['hand'][0]) + (points[:, :, -1] == SEGMENT_ID['object'][0])) > 0
        valid = depth_mask * s_mask

        # get valid point_nums for each env (num_envs,)
        now, point_nums, points_list = 0, valid.sum(dim=1), []
        # (num_envs, num_valid_pts_total, 4)
        valid_points = points[valid]
        
        # presample, make num_pts equal for each env
        for env_id, point_num in enumerate(point_nums):
            if point_num == 0:
                points_list.append(torch.zeros(self.num_pc_presample, valid_points.shape[-1]).to(self.device))
            else:
                # print('env{}_____point_num = {}_____'.format(env_id, point_num))
                points_all = valid_points[now : now + point_num]
                random_ids = torch.randint(0, points_all.shape[0], (self.num_pc_presample,), device=self.device, dtype=torch.long)
                points_all_rnd = points_all[random_ids]
                points_list.append(points_all_rnd)
                now += point_num
        
        # (num_envs, num_pc_presample)
        points_batch = torch.stack(points_list)

        # clean points
        def clean_points(points):
            if torch.sum(points[..., -1] == 0) == 0: return points
            # locate target points
            indices = torch.nonzero(points[..., -1] == 0)
            # change target points
            for n in range(indices.shape[0]):
                if torch.sum(points[indices[n][0], :, -1] != 0) == 0: continue
                points[indices[n][0]][indices[n][1]] = points[indices[n][0]][points[indices[n][0], :, -1] != 0][0]
            return points
        
        # render scene points
        points_fps, _ = sample_farthest_points(points_batch, K=self.num_pc_downsample*2 if render_scene_only else self.num_pc_downsample)
        # render hand and object points
        if not render_scene_only:
            # sample points with target sample_num
            num_sample_dict = self.cfg['env']['vision']['pointclouds']['numSample']
            zeros = torch.zeros((batch_size, self.num_pc_presample), device=self.device).to(torch.long)
            idx = torch.arange(batch_size * self.num_pc_presample, device=self.device).view(batch_size, self.num_pc_presample).to(torch.long)
            # mask first point
            points_batch[0, 0, :] *= 0.
            # extract hand, object points
            hand_idx = torch.where(points_batch[:, :, -1] == SEGMENT_ID[0], idx, zeros)
            hand_pc = points_batch.view(-1, points_batch.shape[-1])[hand_idx]
            object_idx = torch.where(points_batch[:, :, -1] == SEGMENT_ID[0], idx, zeros)
            object_pc = points_batch.view(-1, points_batch.shape[-1])[object_idx]
            # sample hand, object points
            hand_fps, _ = sample_farthest_points(hand_pc, K=num_sample_dict['hand'])
            object_fps, _ = sample_farthest_points(object_pc, K=num_sample_dict['object'])
            # clean hand, object points
            hand_fps = clean_points(hand_fps)
            object_fps = clean_points(object_fps)
            # concat hand, object points
            points_fps = torch.cat([points_fps, hand_fps, object_fps], dim=1)

        # # repose points_fps
        # if self.repose_z: points_fps[..., :3] = self.unpose_pc(points_fps[..., :3])

        # others
        others = {}
        return points_fps, others
    