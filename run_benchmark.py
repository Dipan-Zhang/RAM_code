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
  
def get_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

CFGS_DIR = 'run_realworld/configs'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='all', help='task name')
    parser.add_argument('-n', '--num_trial', type=int, default=5, help='number of trials')
    parser.add_argument('--same_trial_name', action='store_true', default=False, help='whether use same name, for batch eval')

    args = parser.parse_args()

    if args.task == 'all':
        TASKS = PORTABLE_TASK_LIST + ARTICULATE_TASK_LIST
    elif args.task == 'portable':
        TASKS = PORTABLE_TASK_LIST
    elif args.task == 'articulate':
        TASKS = ARTICULATE_TASK_LIST
    # elif args.task == 'drawer':
    #     TASKS = ['open_drawer', 'close_drawer']
    # elif args.task == 'dishwasher':
    #     TASKS = ['open_dishwasher']
    # elif args.task == 'microwave':
    #     TASKS = ['open_microwave', 'close_microwave']
    elif args.task == 'rest':
        TASKS = ['open_cabinet', 'close_cabinet', 'open_dishwasher']
    else:
        TASKS = [args.task]

    for task in tqdm(TASKS):
        # task_env, task_name = task.split("@")
        task_cfg_file = os.path.join(CFGS_DIR, f'{task}.yaml')

        cmd = f"python run_realworld/run.py --config configs/{task}.yaml --retrieve --num_trial {args.num_trial}"
        if args.same_trial_name:
            trial_name = f'trial_{get_time()}'
            cmd += f" --save_dir {trial_name}"
        os.system(cmd)
        print("=======================================================")