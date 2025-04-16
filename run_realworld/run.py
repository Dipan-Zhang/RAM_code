import os
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
import pickle
import tqdm

def get_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

def backup(args, cfgs):
    shutil.copyfile(f"run_realworld/{args.config}", f"{cfgs['SAVE_ROOT']}/config.yaml")

def underscore_string_to_camel_case(string):
    """
    Convert a string from underscore format to camel case format.
    For example, 'my_variable_name' becomes 'MyVariableName'.
    """
    components = string.split('_')
    return ''.join(x.title() for x in components) 

def reverse_object_action(task_name):
    "bottle_open -> open_bottle"
    obj = task_name.split('_')[0]
    action = task_name.split('_')[1:]
    return f"{obj}_{'_'.join(action[::-1])}"

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    cfgs = read_yaml_config(f"run_realworld/{args.config}")
    torch.set_printoptions(precision=4, sci_mode=False)
    task_name = args.config.split('/')[-1].split('.')[0]
            
    instruction = cfgs['instruction']
    obj = cfgs['obj']
    prompt = cfgs['prompt']
    data_source = cfgs.get("DATA_SOURCE", "droid")

    # save_root = cfgs['SAVE_ROOT']
    RLbench_task_name = underscore_string_to_camel_case(task_name)
    if args.save_dir == '':
        SAVE_ROOT = os.path.join('../RLBench/outputs', RLbench_task_name, 'RAM', f'trial_{get_time()}')
    else:
        # SAVE_ROOT = args.save_dir
        SAVE_ROOT = os.path.join('../RLBench/outputs', RLbench_task_name, 'RAM', args.save_dir)
    cfgs['SAVE_ROOT'] = SAVE_ROOT
    os.makedirs(SAVE_ROOT, exist_ok=True)
    backup(args, cfgs)
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")


    results_all = {}
    dataset_path = f'../RLBench/outputs/{RLbench_task_name}/obs/'
    camera_names = os.listdir(dataset_path)
    for camera_name in camera_names:
        results_per_camera = {}
        dataset_path_cam = os.path.join(dataset_path, camera_name)
        for trial in tqdm.tqdm(range(args.num_trial), desc=f"{RLbench_task_name, camera_name}"):
            SAVE_ROOT_TRIAL = os.path.join(SAVE_ROOT, camera_name, f"trial_{trial}")
            cfgs['SAVE_ROOT'] = SAVE_ROOT_TRIAL
            gym = MiniEnv(cfgs, grounded_dino_model, sam_predictor)
            os.makedirs(SAVE_ROOT_TRIAL, exist_ok=True)
            subset_retrieve_pipeline = SubsetRetrievePipeline(
                                        subset_dir="assets/data",
                                        save_root=SAVE_ROOT_TRIAL,
                                        lang_mode='clip',
                                        topk=5, 
                                        crop=True,
                                        data_source=data_source,
                                        )
            # input_dir = f"run_realworld/real_data/input/{obj}"
            pcd = o3d.io.read_point_cloud(os.path.join(dataset_path_cam, "pcd.ply"))
            rgb = Image.open(os.path.join(dataset_path_cam, "rgb.png"))

            tgt_img_PIL = rgb
            tgt_img_PIL.save(f"{SAVE_ROOT_TRIAL}/tgt_img.png")
            rgb = np.array(rgb)
            
            tgt_masks = inference_one_image(np.array(tgt_img_PIL), grounded_dino_model, sam_predictor, box_threshold=cfgs['box_threshold'], text_threshold=cfgs['text_threshold'], text_prompt=obj, device="cuda").cpu().numpy() # you can set point_prompt to traj[0]
            tgt_mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            # if mask is false, make it white
            tgt_img_masked = np.array(tgt_img_PIL) * tgt_mask + 255 * (1 - tgt_mask)
            # tgt_img_masked, _, _ = crop_image(tgt_img_masked, tgt_mask)
            tgt_img_PIL = Image.fromarray(tgt_img_masked).convert('RGB')
            tgt_img_PIL.save(f"{SAVE_ROOT_TRIAL}/tgt_img_masked.png")
            
            ######## src
            ####################### SOURCE DEMONSTRATION ########################
            if not args.retrieve:
                data_dict = np.load("run_realworld/real_data/demonstration/data.pkl", allow_pickle=True)
                traj = data_dict['traj']
                src_img_np = data_dict['masked_img']
                src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
                src_img_PIL.save(f"{SAVE_ROOT_TRIAL}/src_img.png")
            else:
                # use retrieval to get src_path (or src image) and src trajectory in 2d space
                _, top1_retrieved_data_dict = subset_retrieve_pipeline.retrieve(instruction, np.array(tgt_img_PIL))
                traj = top1_retrieved_data_dict['traj'] # 2D
                src_img_np = top1_retrieved_data_dict['masked_img']
                src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
                src_img_PIL.save(f"{SAVE_ROOT_TRIAL}/src_img.png")

            # scale cropped_traj to IMG_SIZE
            src_pos_list = []
            for xy in traj: # xy: (x, y)
                src_pos_list.append((xy[0] * IMG_SIZE / src_img_PIL.size[0], xy[1] * IMG_SIZE / src_img_PIL.size[1]))
            
            while True:
                try:
                    contact_point, post_contact_dir = transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=SAVE_ROOT_TRIAL, ftype='sd')
                    break
                except Exception as transfer_e:
                    traceback.print_exc()
                    print('[ERROR] in transfer_affordance:', transfer_e)

            # contact point + post-contact direction
            ret_dict = gym.lift_affordance(rgb, pcd, contact_point, post_contact_dir)
            
            np.savez(f"{SAVE_ROOT_TRIAL}/RAM_ret_dict_{trial}.npz", **ret_dict)
            print("3D Affordance:\n", ret_dict)
            results_per_camera[trial] = ret_dict
            del subset_retrieve_pipeline
        results_all[camera_name] = results_per_camera
    # save all the results
    with open(os.path.join(SAVE_ROOT, f"retrieved_motion_all.pkl"), 'wb') as f:
        pickle.dump(results_all, f)
    
    print("====== DONE ======")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to the config file') # e.g. configs/drawer_open.yaml
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--retrieve', action='store_true')
    parser.add_argument('-n', '--num_trial', type=int, default=5)
    parser.add_argument('-s', '--save_dir', type=str, default='')
    args = parser.parse_args()
    
    main(args)

# check run_task.sh for running this scriot