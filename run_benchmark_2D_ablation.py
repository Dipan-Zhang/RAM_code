import numpy as np
import os
from tqdm import tqdm
import argparse



PORTABLE_TASK_LIST=[
    'pick_up_cup', 
    'pick_up_bottle',
    'pick_up_mug',
    'pick_up_bowl',
  ]

ARTICULATE_TASK_LIST=[
'open_drawer', 
'close_drawer',
'open_microwave',
'close_microwave',
'open_cabinet',
'close_cabinet',
'open_dishwasher',

'close_laptop',
'down_toilet_seat',
'open_slide_cabinet',
'close_slide_cabinet',
# 'open_fridge', 
# 'open_washing_machine',
]
  
#################### THIS SCRIPT HAS TO BE RUN IN SLURM because of memory ######################
def get_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

def underscore_string_to_camel_case(string):
    """
    Convert a string from underscore format to camel case format.
    For example, 'my_variable_name' becomes 'MyVariableName'.
    """
    components = string.split('_')
    return ''.join(x.title() for x in components) 

CFGS_DIR = 'run_realworld/configs'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='2D_ablation', help='task name')
    parser.add_argument('-n', '--num_trial', type=int, default=5, help='number of trials')
    parser.add_argument('--num_var', type=int, default=0, help='use variation')
    args = parser.parse_args()


    if args.task == '2D_ablation':
        TASKS = [
            # 'open_microwave',
            # 'close_microwave',
            # 'open_drawer',
            # 'open_cabinet',
            # 'open_dishwasher',
            'open_slide_cabinet',
            # 'close_laptop',
        ]
    else:
        TASKS = [args.task]

    trial_name = f'trial_{get_time()}'

    SAVE_BASE_DIR = '../RLBench/outputs_ablation_2D'

    for task in tqdm(TASKS):
        print(f"==================TASKS: {task}========================")
        task_cfg_file = os.path.join(CFGS_DIR, f'{task}.yaml')
        taskName = underscore_string_to_camel_case(task)
        if args.num_var==0:
            SAVE_ROOT = os.path.join(SAVE_BASE_DIR, taskName, 'RAM', trial_name)
            os.makedirs(SAVE_ROOT, exist_ok=True)
            cmd = f"python run_realworld/run.py --config configs_2D_ablation/{task}.yaml --retrieve --num_trial {args.num_trial} --save_dir {SAVE_ROOT}"
            os.system(cmd)
        else: 
            for var in range(args.num_var):
                SAVE_ROOT = os.path.join(SAVE_BASE_DIR, taskName, 'RAM', trial_name, f'var_{var}')
                os.makedirs(SAVE_ROOT, exist_ok=True)
                cmd = f"python run_realworld/run_var.py --config configs/{task}.yaml --retrieve --num_trial {args.num_trial} --save_dir {SAVE_ROOT}" 
                os.system(cmd)
        print("=======================================================")