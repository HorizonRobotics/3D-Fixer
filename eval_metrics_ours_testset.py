# SPDX-FileCopyrightText: 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import json
import torch
import random
import imageio
import numpy as np
import fsspec
import open3d as o3d
import copy
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm

def get_topN_instant_mask(instance_mask_path, n=5):
    assert n > 0
    index_mask = imageio.v3.imread(instance_mask_path)
    h, w = index_mask.shape[:2]
    index_mask = np.rint(index_mask.astype(np.float32) / 65535 * 100.0) # hand coded, max obj nums = 100
    instance_list = np.unique(index_mask).astype(np.uint8)
    instance_list = instance_list[instance_list != 0]
    instance_list = instance_list[instance_list != 1]
    instance_list = instance_list[instance_list != 2]

    mask_pack = []
    mask_ratio = []
    for inst_idx in instance_list:
        mask_pack.append(
            index_mask == inst_idx
        )
        mask_ratio.append(
            mask_pack[-1].astype(np.float32).sum() / (h * w)
        )

    mask_ratio = np.array(mask_ratio)
    mask_ratio_ind = np.argsort(mask_ratio)[-n:]

    selected_mask_packs = []
    selected_mask_idxs = []
    for idx in mask_ratio_ind:
        selected_mask_packs.append(mask_pack[idx])
        selected_mask_idxs.append(instance_list[idx])

    return selected_mask_packs, selected_mask_idxs

def voxelize(points, voxel_size, grid_size, min_bound):
    """
    Reference: https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/utils/metrics.py#L94
    Voxelize the point cloud.

    Args:
        points (torch.Tensor): Point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        min_bound (torch.Tensor): Minimum bounds of the grid

    Returns:
        torch.Tensor: Voxel grid of shape (B, grid_size, grid_size, grid_size) with boolean type
    """
    # Shift points to positive grid and scale
    scaled_points = (points - min_bound) / voxel_size
    indices = torch.floor(scaled_points).long()

    # Clamp indices to grid size
    indices = indices.clamp(0, grid_size - 1)

    # Create voxel grid
    voxel_grid = torch.zeros(
        (points.shape[0], grid_size, grid_size, grid_size),
        device=points.device,
        dtype=torch.bool,
    )
    for b in range(points.shape[0]):
        voxel_grid[b, indices[b, :, 0], indices[b, :, 1], indices[b, :, 2]] = True

    return voxel_grid

def compute_volume_iou(pred, gt, voxel_size=0.05, grid_size=64, mode="bbox"):
    """
    Reference: https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/utils/metrics.py#L126
    Compute Volume IoU between predicted and ground truth point clouds.

    Args:
        pred (torch.Tensor): Predicted point cloud of shape (B, N, 3)
        gt (torch.Tensor): Ground truth point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        mode (str): Mode of computing volume iou, either "pcd" or "bbox"

    Returns:
        torch.Tensor: Volume IoU for each batch
    """
    if mode == "pcd":
        # Define the grid bounds
        min_bound = torch.min(torch.min(pred, dim=1).values, dim=0).values
        max_bound = torch.max(torch.max(pred, dim=1).values, dim=0).values
        min_bound = torch.min(min_bound, torch.min(gt, dim=1).values.min(dim=0).values)
        max_bound = torch.max(max_bound, torch.max(gt, dim=1).values.max(dim=0).values)

        # Voxelize the point clouds
        pred_voxels = voxelize(pred, voxel_size, grid_size, min_bound)
        gt_voxels = voxelize(gt, voxel_size, grid_size, min_bound)

        # Compute intersection and union
        intersection = (pred_voxels & gt_voxels).sum(dim=(1, 2, 3)).float()
        union = (pred_voxels | gt_voxels).sum(dim=(1, 2, 3)).float()

        # Compute IoU
        iou = intersection / (union + 1e-8)

    elif mode == "bbox":
        # Compute bounding boxes
        pred_min = pred.min(dim=1).values
        pred_max = pred.max(dim=1).values
        gt_min = gt.min(dim=1).values
        gt_max = gt.max(dim=1).values

        # Compute intersection
        intersection_min = torch.max(pred_min, gt_min)
        intersection_max = torch.min(pred_max, gt_max)
        inter_dims = (intersection_max - intersection_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

        # Compute union
        pred_dims = (pred_max - pred_min).clamp(min=0)
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]  # (B,)
        gt_dims = (gt_max - gt_min).clamp(min=0)
        gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]

        # Compute IoU
        union_vol = pred_vol + gt_vol - inter_vol
        iou = inter_vol / (union_vol + 1e-8)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return iou

def compute_scene_level_metrics(all_pred_vertices, all_gt_vertice, eps=1e-6):
    gt_vertices = o3d.geometry.PointCloud()
    gt_vertices.points = o3d.utility.Vector3dVector(
        np.concatenate(
            [np.array(verts.points).astype(np.float32) for verts in all_gt_vertice]
        )
    )
    pred_vertices = o3d.geometry.PointCloud()
    pred_vertices.points = o3d.utility.Vector3dVector(
        np.concatenate(
            [np.array(verts.points).astype(np.float32) for verts in all_pred_vertices]
        )
    )

    # Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L71
    dist_gt_pred = np.array(gt_vertices.compute_point_cloud_distance(pred_vertices))
    dist_pred_gt = np.array(pred_vertices.compute_point_cloud_distance(gt_vertices))

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    metrics = {
        'Chamfer': (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return metrics

def compute_object_level_metrics(pred_vertices, gt_vertices, gt_trans, gt_scale, gt_rot, eps=1e-6):
    pred_vertices = copy.deepcopy(pred_vertices)
    gt_vertices = copy.deepcopy(gt_vertices)

    pred_vertices_pt = torch.from_numpy(np.array(pred_vertices.points).astype(np.float32))
    gt_vertices_pt = torch.from_numpy(np.array(gt_vertices.points).astype(np.float32))

    iou = compute_volume_iou(pred_vertices_pt.unsqueeze(0), gt_vertices_pt.unsqueeze(0)).item()

    pred_vertices.translate(gt_trans)
    pred_vertices.scale(1. / gt_scale, center=(0., 0., 0.))
    R1 = pred_vertices.get_rotation_matrix_from_xyz((0, 0, -gt_rot[2]))
    pred_vertices.rotate(R1, center=(0., 0., 0.))
    pred_vertices.scale(0.95 / 0.5, center=(0., 0., 0.))
    
    gt_vertices.translate(gt_trans)
    gt_vertices.scale(1. / gt_scale, center=(0., 0., 0.))
    R1 = gt_vertices.get_rotation_matrix_from_xyz((0, 0, -gt_rot[2]))
    gt_vertices.rotate(R1, center=(0., 0., 0.))
    gt_vertices.scale(0.95 / 0.5, center=(0., 0., 0.))

    # Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L71
    dist_gt_pred = np.array(gt_vertices.compute_point_cloud_distance(pred_vertices))
    dist_pred_gt = np.array(pred_vertices.compute_point_cloud_distance(gt_vertices))

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    metrics = {
        'IoU': iou,
        'Chamfer': (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return metrics

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the generated results.')
    parser.add_argument('--testset_dir', type=str, required=True,
                        help='Directory to load the testset scenes.')
    parser.add_argument('--assets_dir', type=str, required=True,
                        help='Directory to load the 3D assets.')
    parser.add_argument('--top_instance_mask', type=int, default=5,
                        help='The top N instances according to valid pixel ratio.')
    parser.add_argument('--seed', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    set_random_seed(opt.seed)

    testset_dir = opt.testset_dir
    if os.path.exists(os.path.join(testset_dir, 'transforms_list.json')):
        with open(os.path.join(testset_dir, 'transforms_list.json'), 'r') as f:
            transforms_list = json.load(f)
    else:
        fs, path = fsspec.core.url_to_fs(testset_dir)
        transforms_list = sorted(fs.glob(
            testset_dir + f"/*/transforms.json"
        ))
        with open(os.path.join(testset_dir, 'transforms_list.json'), 'w') as f:
            json.dump(transforms_list, f)
    transforms_list = sorted(transforms_list)

    eval_metrics = {}
    all_obj_metrics = {
        "IoU": [],
        "Chamfer": [],
        "F1": [],
    }
    all_scene_metrics = {
        "Chamfer": [],
        "F1": [],
    }
    for transforms_path in tqdm(transforms_list):

        with open(transforms_path, 'r') as f:
            js = json.load(f)

        selected_frame = js['frames'][0]
        scene_sha256 = transforms_path.split('/')[-2]
        scene_root_dir = os.path.dirname(transforms_path)
        instances_gt = js['instance']

        save_dir = os.path.join(opt.output_dir, scene_sha256)
        save_mesh_root = os.path.join(save_dir, 'objects')
        instance_mask_path = os.path.join(scene_root_dir, selected_frame['file_path'].replace('.png', '_index.png'))
        mask_pack, mask_pack_inst_idx = get_topN_instant_mask(instance_mask_path, n=opt.top_instance_mask)

        scene_metrics = {'sha256': scene_sha256}
        pred_vertices = []
        gt_vertices = []
        num_points = 20480
        for object_id, object_mask in zip(mask_pack_inst_idx, mask_pack):
            try:
                instance_name = f'{object_id}'

                if not os.path.exists(os.path.join(save_mesh_root, f'{instance_name}.glb')):
                    continue

                instance_sha256 = instances_gt[f'{object_id}']['sha256']
                if not os.path.exists(os.path.join(opt.assets_dir, instance_sha256, 'mesh.ply')):
                    continue

                instance_sha256 = instances_gt[f'{object_id}']['sha256']
                gt_rot = instances_gt[f'{object_id}']['rand_rot']
                gt_scale = instances_gt[f'{object_id}']['rand_scale']
                gt_trans = np.array(instances_gt[f'{object_id}']['rand_trans'])
                gt_mesh = o3d.io.read_triangle_mesh(os.path.join(opt.assets_dir, instance_sha256, 'mesh.ply'))
                R1 = gt_mesh.get_rotation_matrix_from_xyz((0, 0, gt_rot[2]))
                gt_mesh.rotate(R1, center=(0., 0., 0.))
                gt_mesh.scale(gt_scale, center=(0., 0., 0.))
                gt_mesh.translate(-gt_trans)

                gt_vertices.append(
                    gt_mesh.sample_points_uniformly(num_points)
                )

                pred_vertices.append(
                    o3d.io.read_triangle_mesh(os.path.join(save_mesh_root, f'{instance_name}.glb')).sample_points_uniformly(num_points)
                )

                obj_metrics = compute_object_level_metrics(pred_vertices[-1],
                                                           gt_vertices[-1],
                                                           gt_trans, gt_scale, 
                                                           gt_rot)
                scene_metrics[instance_name] = obj_metrics
                all_obj_metrics['IoU'].append(obj_metrics['IoU'])
                all_obj_metrics['Chamfer'].append(obj_metrics['Chamfer'])
                all_obj_metrics['F1'].append(obj_metrics['F1'])

            except Exception as e:
                print (instance_name, e)

        try:
            if len(pred_vertices) > 0:
                scene_metrics['scene'] = compute_scene_level_metrics(pred_vertices, gt_vertices)
                eval_metrics[scene_sha256] = scene_metrics
                with open(os.path.join(save_dir, "scene_metrics.json"), 'w') as f:
                    json.dump(scene_metrics, f)
                all_scene_metrics['Chamfer'].append(scene_metrics['scene']['Chamfer'])
                all_scene_metrics['F1'].append(scene_metrics['scene']['F1'])

        except Exception as e:
            print (f'{instance_name}: {e}')

    eval_metrics['mean'] = {
        "CD-S": np.mean(all_scene_metrics['Chamfer']),
        "F-S": np.mean(all_scene_metrics['F1']),
        "CD-O": np.mean(all_obj_metrics['Chamfer']),
        "F-O": np.mean(all_obj_metrics['F1']),
        "IoU-B": np.mean(all_obj_metrics['IoU']),
    }

    with open(os.path.join(opt.output_dir, f'all_metrics.json'), 'w') as f:
        json.dump(eval_metrics, f)
    