import imageio
import open3d as o3d
import cv2
import math
import numpy as np
import torch
import random
import time
from run_realworld.utils import get_bounding_box, crop_points, cluster_normals, visualize_point_directions
import os, json
import sys
import cv2
sys.path.append("../")
sys.path.append("vision")
# from vision.GroundedSAM.grounded_sam_utils import prepare_GroundedSAM_for_inference, inference_one_image
from graspness_implementation.gsnet import GSNet, grasp_to_pointcloud, vis_save_grasp, get_best_grasp, get_pose_from_grasp, get_closest_grasp, get_default_grasp
import argparse
import traceback
from matplotlib import pyplot as plt
from run_realworld.utils import read_yaml_config
import ipdb


class MiniEnv():
    def __init__(
            self, 
            cfgs=None,
            grounded_dino_model = None, 
            sam_predictor = None,
        ):
        self.cfgs = cfgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam_w = cfgs['cam_w']
        self.cam_h = cfgs['cam_h']
        
        # if grounded_dino_model is not None and sam_predictor is not None:
        #     self.grounded_dino_model = grounded_dino_model
        #     self.sam_predictor = sam_predictor
        #     self.box_threshold = cfgs['box_threshold']
        #     self.text_threshold = cfgs['text_threshold']
        # elif self.cfgs["INFERENCE_GSAM"]:
        #     self.prepare_groundedsam()
        if self.cfgs["USE_GSNET"]:
            self.prepare_gsnet()
            
    # def prepare_groundedsam(self):
    #     self.box_threshold = self.cfgs['box_threshold']
    #     self.text_threshold = self.cfgs['text_threshold']
    #     sam_version = "vit_h"
    #     sam_checkpoint = "assets/ckpts/sam_vit_h_4b8939.pth"
    #     grounded_checkpoint = "assets/ckpts/groundingdino_swint_ogc.pth"
    #     config = "vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    #     self.grounded_dino_model, self.sam_predictor = prepare_GroundedSAM_for_inference(sam_version=sam_version, sam_checkpoint=sam_checkpoint,
    #             grounded_checkpoint=grounded_checkpoint, config=config, device=self.device)

    def prepare_gsnet(self):
        self.gsnet = GSNet(self.cfgs["gsnet"])
    
    def inference_gsnet(self, pcs, keep=1e6, nms=True):
        gg = self.gsnet.inference(pcs)
        if nms:
            gg = gg.nms()
        gg = gg.sort_by_score()
        if len(gg) > keep:
            gg = gg[:keep]
        if self.cfgs["gsnet"]["vis"]:
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers]) 
            
            # pcd_w_grasp = grasp_to_pointcloud(grippers, cloud)
            # o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/pcd_w_grasp.ply", pcd_w_grasp)
        
        return gg
    
    def detect_grasp_gsnet(self, points, colors=None, save_vis=False):
        '''GSNet'''
        # need to preprocess point cloud
        pcs_input = points.copy()
        pcs_input[...,2] = -pcs_input[...,2]
        gg = self.inference_gsnet(pcs_input, nms=False)
        print(gg[0])
        # adjust grasps
        for g_i in range(len(gg)):
            translation = gg[g_i].translation
            rotation = gg[g_i].rotation_matrix
            translation = np.array([translation[0], translation[1], -translation[2]])
            rotation[2, :] = -rotation[2, :]
            rotation[:, 2] = -rotation[:, 2]
            gg.grasp_group_array[g_i][13:16] = translation
            gg.grasp_group_array[g_i][4:13] = rotation.reshape(-1)
        # print(gg[0])
        print('grasp num:', len(gg))
        if save_vis:
            vis_save_grasp(points, gg, f"{self.cfgs['SAVE_ROOT']}/gsnet.ply")
        return gg

    ### affordance
    def lift_affordance(self, rgb, pcd, pixel, dir):
        post_contact_dirs_2d, post_contact_dirs_3d = None, None
        partial_points = np.array(pcd.points)
        partial_colors = np.array(pcd.colors)
        position = partial_points[pixel[1]*self.cam_w + pixel[0]] # lift 2d pixel to 3d position
        
        ipdb.set_trace()
        # visualization
        # ds_points, _, _ = get_downsampled_pc(partial_points, None, 20000)
        ds_points, _, _ = crop_points(position, partial_points, thres=0.5) # search points within cube (r=thres[m])
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(ds_points)
        # red
        save_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((ds_points.shape[0], 3)))
        # add a point
        save_pcd.points.append(position)
        save_pcd.colors.append(np.array([0, 1, 0]))
        # save to ply
        o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/grasp_point.ply", save_pcd)
        
        MAX_ATTEMPTS = 20 # in case there is no good grasp at one time
        max_dis = 0.05
        best_grasp = None
        max_radius, min_radius = 0.2, 0.1
        gg = None
        for num_attempt in range(MAX_ATTEMPTS):
            try:
                # get a smaller crop radius along trial
                crop_radius = max_radius - (max_radius - min_radius) * num_attempt / MAX_ATTEMPTS
                print('=> crop_radius:', crop_radius)
                cropped_points, cropped_colors, cropped_normals = crop_points(
                    position, partial_points, partial_colors, thres=crop_radius, save_root=self.cfgs['SAVE_ROOT']
                )
                try:
                    # gg = self.detect_grasp_anygrasp(cropped_points, cropped_colors, save_vis=True) # use AnyGrasp if properly set up
                    gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, save_vis=True)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    traceback.print_exc()
                if gg is None:
                    continue
                print('=> total grasp:', len(gg))
                if len(gg) == 0:
                    continue
                
                best_grasp = get_best_grasp(gg, position, max_dis=max_dis) # original: 0.03
                if best_grasp is None:
                    print('==>> no best grasp')
                else:
                    break
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
        # if still no best -> use cloest grasp
        if best_grasp is None:
            try:
                gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, False)
            except:
                gg = self.detect_grasp_gsnet(partial_points, partial_colors, False)
            best_grasp = get_closest_grasp(gg, position)
            print('==>> use GSNet for closest grasp')
        n_clusters = 5
        vis_save_grasp(cropped_points, best_grasp, f"{self.cfgs['SAVE_ROOT']}/best_grasp.ply")
        clustered_centers = cluster_normals(cropped_normals, n_clusters=n_clusters) # (2*n_clusters, 3)
        visualize_point_directions(cropped_points, position, clustered_centers, self.cfgs['SAVE_ROOT'])
        # post_contact_dirs_3d = clustered_centers
        # post_contact_dirs_2d = self.project_normals(rgb, pixel, clustered_centers)
        grasp_array = best_grasp.grasp_array.tolist()
        
        # post-grasp
        best_dir_3d, best_score = None, -1
        # for i in range(post_contact_dirs_2d.shape[0]):
        #     score = np.dot(post_contact_dirs_2d[i], dir)
        #     if score > best_score:
        #         best_score = score
        #         best_dir_3d = post_contact_dirs_3d[i]
        # visualize_point_directions(ds_points, position, [best_dir_3d], self.cfgs['SAVE_ROOT'], "best_dir_3d")
        # post_grasp_dir = best_dir_3d.tolist()
        
        ret_dict = {
            "grasp_array": grasp_array,
            # "post_grasp_dir": post_grasp_dir
        }
        return ret_dict
    
    def get_best_grasp(self, pcd, pixel, dir):
        post_contact_dirs_2d, post_contact_dirs_3d = None, None
        partial_points = np.array(pcd.points)
        partial_colors = np.array(pcd.colors)
        position = partial_points[pixel[1]*self.cam_w + pixel[0]] # lift 2d pixel to 3d position
        
        # visualization
        # ds_points, _, _ = get_downsampled_pc(partial_points, None, 20000)
        ds_points, _, _ = crop_points(position, partial_points, thres=0.5) # search points within cube (r=thres[m])
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(ds_points)
        # red
        save_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((ds_points.shape[0], 3)))
        # add a point
        save_pcd.points.append(position)
        save_pcd.colors.append(np.array([0, 1, 0]))
        # save to ply
        o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/grasp_point.ply", save_pcd)
        
        MAX_ATTEMPTS = 20 # in case there is no good grasp at one time
        max_dis = 0.05
        best_grasp = None
        max_radius, min_radius = 0.2, 0.1
        gg = None # grasp group
        for num_attempt in range(MAX_ATTEMPTS):
            try:
                # generate grasp group (gg) around within crop radius
                crop_radius = max_radius - (max_radius - min_radius) * num_attempt / MAX_ATTEMPTS
                print('=> crop_radius:', crop_radius)
                cropped_points, cropped_colors, cropped_normals = crop_points(
                    position, partial_points, partial_colors, thres=crop_radius, save_root=self.cfgs['SAVE_ROOT']
                )
                try:
                    # gg = self.detect_grasp_anygrasp(cropped_points, cropped_colors, save_vis=True) # use AnyGrasp if properly set up
                    gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, save_vis=True)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    traceback.print_exc()
                if gg is None:
                    continue
                print('=> total grasp:', len(gg))
                if len(gg) == 0:
                    continue
                
                # select best grasp around selected pixel
                best_grasp = get_best_grasp(gg, position, max_dis=max_dis) # original: 0.03
                if best_grasp is None:
                    print('==>> no best grasp')
                else:
                    break
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()

            print(num_attempt)
        # if still no best -> use cloest grasp
        if best_grasp is None:
            try:
                gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, False)
            except:
                gg = self.detect_grasp_gsnet(partial_points, partial_colors, False)
            best_grasp = get_closest_grasp(gg, position)
            print('==>> use GSNet for closest grasp')

        return best_grasp
    
def pick_points_in_viewer(points, scene_colors=None, verbose=False) -> np.ndarray:
    def pick_points(pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if scene_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    elif isinstance(points, o3d.cuda.pybind.geometry.TriangleMesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = points.vertices
        pcd.colors = points.vertex_colors
    else:
        pcd = points

    picked_ids = pick_points(pcd)
    final_points = np.asarray(pcd.points)[picked_ids]
    print("Points selected: ", final_points.shape, final_points)

    if verbose:
        print("Final points: ")
        for i in range(len(final_points)):
            print(final_points[i])
    return final_points

if __name__ == '__main__':
    from PIL import Image
    import pdb
    import matplotlib.pyplot as plt
    cfgs = read_yaml_config(f"run_realworld/configs/drawer_open.yaml")
    os.makedirs(cfgs['SAVE_ROOT'], exist_ok=True)
    gym = MiniEnv(cfgs)

    pcd = o3d.io.read_point_cloud("run_realworld/real_data/input/pcd.ply")
    rgb = Image.open("run_realworld/real_data/input/rgb.png")
    
    # click 2d points
    def choose_pixel(rgb):
        points = []
        labels = []
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(rgb)
        def onclick(event):
            if event.button == 1:  # Left click for positive point
                points.append([event.xdata, event.ydata])
                labels.append(1)
                ax.scatter(event.xdata, event.ydata, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
            elif event.button == 3:  # Right click for negative point
                points.append([event.xdata, event.ydata])
                labels.append(0)
                ax.scatter(event.xdata, event.ydata, color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        points = np.array(points, dtype=np.int32)
        return points
    pixel = choose_pixel(rgb)
    post_contact_dir = np.array([0, 0, -1])
    # ret_dict = gym.lift_affordance(rgb, pcd, pixel.reshape(2,), post_contact_dir)
    best_grasp = gym.get_best_grasp(pcd, pixel.reshape(2,), post_contact_dir)
    print(best_grasp)
    ipdb.set_trace()
