import os
import numpy as np
import argparse
# from affordance.helpers import load_pickle, save_pickle
import pickle

if __name__ == '__main__':
    "compose subresults from mutiple cameras together into transferred_motion_all.pkl for RLbench to test"
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial_dir', type=str)
    parser.add_argument('-n', '--num_trial', type=int, default=5, help='number of trials')
    args = parser.parse_args()

    trial_dir = args.trial_dir
    assert os.path.exists(trial_dir), f"Directory {trial_dir} does not exist."
    base_dir = '/'.join(trial_dir.split('/')[:-2])
    obs_dir = os.path.join(base_dir, 'obs')
    obs_list = os.listdir(obs_dir)

    results_all = {}
    for obs in obs_list:
        results_per_camera = {}
        for trial in range(args.num_trial):
            results_fn = os.path.join(trial_dir, obs, f'trial_{trial}', f'RAM_ret_dict_{trial}.npz')
            results = np.load(results_fn, allow_pickle=True)
            results_per_camera[trial] = dict(results)
        results_all[obs] = results_per_camera

    with open(os.path.join(trial_dir, f"retrieved_motion_all.pkl"), 'wb') as f:
        pickle.dump(results_all, f)
    print(f"Saved all results to {os.path.join(trial_dir, f'retrieved_motion_all.pkl')}")
    print("====== DONE ======")
