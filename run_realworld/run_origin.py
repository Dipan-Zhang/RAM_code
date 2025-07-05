import os
import sys
# Add the project root to Python path so imports work from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
from run_realworld.env import MiniEnv
import numpy as np
from run_realworld.utils import read_yaml_config
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
import torch
from PIL import Image
import glob, time
from vision.featurizer.run_featurizer import transfer_affordance
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline import SubsetRetrievePipeline
import argparse
import traceback
import matplotlib
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use
from tqdm import tqdm
import shutil
import random
import open3d as o3d
import gc

def backup(args, cfgs):
    shutil.copyfile(f"run_realworld/{args.config}", f"{cfgs['SAVE_ROOT']}/config.yaml")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    cfgs = read_yaml_config(f"run_realworld/{args.config}")
    cfgs['SAVE_ROOT'] = args.save_dir
    os.makedirs(cfgs['SAVE_ROOT'], exist_ok=True)
    backup(args, cfgs)
    torch.set_printoptions(precision=4, sci_mode=False)
            
    instruction = cfgs['instruction']
    obj = cfgs['obj']
    prompt = cfgs['prompt']
    data_source = cfgs.get("DATA_SOURCE", "droid")
    save_root = cfgs['SAVE_ROOT']
    
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    gym = MiniEnv(cfgs, None, None)
    
    if args.retrieve:
        subset_retrieve_pipeline = SubsetRetrievePipeline(
            subset_dir="assets/data",
            save_root=save_root,
            lang_mode='clip',
            topk=5, 
            crop=True,
            data_source=data_source,
        )
    
    raw_data_dict = np.load(f"{args.data_dir}/raw_input.npz", allow_pickle=True)
    rgb = raw_data_dict['color']
    depth = raw_data_dict['depth']
    intr = raw_data_dict['intr']
    pts = raw_data_dict['pts_undis']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((pts.shape[0], 3)))
    o3d.io.write_point_cloud(f"{save_root}/scene_pcd.ply", pcd)

    tgt_img_PIL = Image.fromarray(rgb)
    tgt_img_PIL.save(f"{save_root}/tgt_img.png")

    tgt_masks = inference_one_image(np.array(tgt_img_PIL), grounded_dino_model, sam_predictor, box_threshold=cfgs['box_threshold'], text_threshold=cfgs['text_threshold'], text_prompt=obj, device="cuda").cpu().numpy() # you can set point_prompt to traj[0]

    tgt_mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    # if mask is false, make it white
    tgt_img_masked = np.array(tgt_img_PIL) * tgt_mask + 255 * (1 - tgt_mask)
    # tgt_img_masked, _, _ = crop_image(tgt_img_masked, tgt_mask)
    tgt_img_PIL = Image.fromarray(tgt_img_masked).convert('RGB') # HW3
    tgt_img_PIL.save(f"{save_root}/tgt_img_masked.png")
    
    # ######## src
    # ####################### SOURCE DEMONSTRATION ########################
    if not args.retrieve:
        retrieve_data_dict = np.load(f"{args.data_dir}/src_data.npz", allow_pickle=True)
        src_pos_list = retrieve_data_dict['src_pos_list']
        src_img_np = retrieve_data_dict['src_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
    else:
        # use retrieval to get src_path (or src image) and src trajectory in 2d space
        _, top1_retrieved_data_dict = subset_retrieve_pipeline.retrieve(instruction, np.array(tgt_img_PIL))
        traj = top1_retrieved_data_dict['traj']
        src_img_np = top1_retrieved_data_dict['masked_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')

        # scale cropped_traj to IMG_SIZE
        src_pos_list = []
        for xy in traj:
            src_pos_list.append((xy[0] * IMG_SIZE / src_img_PIL.size[0], xy[1] * IMG_SIZE / src_img_PIL.size[1]))
        
        # save retrieved src pos list and src img into a npz file
        np.savez(f"{save_root}/src_data.npz", src_pos_list=src_pos_list, src_img=src_img_np)
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
    ####################### SOURCE DEMONSTRATION ########################

    del sam_predictor, grounded_dino_model
    gc.collect()
    torch.cuda.empty_cache()

    while True:
        try:
            contact_point, post_contact_dir = transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=save_root, ftype='sd')
            break
        except Exception as transfer_e:
            traceback.print_exc()
            print('[ERROR] in transfer_affordance:', transfer_e)

    # contact point + post-contact direction
    ret_dict = gym.lift_affordance(rgb, pcd, contact_point, post_contact_dir, depth, intr)
    
    print("3D Affordance:\n", ret_dict)
    # save ret_dict into a npz file
    np.savez(f"{save_root}/ret_dict.npz", **ret_dict)
    
    print("====== DONE ======")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config file') # e.g. configs/drawer_open.yaml
    parser.add_argument('--data_dir', type=str, default='run_realworld/real_data')
    parser.add_argument('--save_dir', type=str, default='run_realworld/gym_outputs')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--retrieve', action='store_true')
    args = parser.parse_args()
    
    main(args)

    # python run_realworld/run_origin.py --config configs_real_exp/open_microwave.yaml --data_dir run_realworld/real_data/real_exp/open_microwave --save_dir run_realworld/gym_outputs/open_microwave 
    # python run_realworld/run_origin.py --config configs_real_exp/close_microwave.yaml --data_dir run_realworld/real_data/real_exp/close_microwave --save_dir run_realworld/gym_outputs/close_microwave