#!/bin/bash 
python run_realworld/run.py --config configs/open_drawer.yaml --retrieve
python run_realworld/run.py --config configs/close_drawer.yaml --retrieve
python run_realworld/run.py --config configs/open_microwave.yaml --retrieve
python run_realworld/run.py --config configs/close_microwave.yaml --retrieve

python run_realworld/run.py --config configs/pick_up_bowl.yaml --retrieve
python run_realworld/run.py --config configs/pick_up_bottle.yaml --retrieve
# python run_realworld/run.py --config configs/pick_up_cup.yaml --retrieve
# python run_realworld/run.py --config configs/pick_up_mug.yaml --retrieve