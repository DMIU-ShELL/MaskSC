import subprocess
import shlex
import time
import os
import argparse

commands_ctgraph_ppo = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py baseline --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py baseline --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py baseline --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py baseline --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py baseline --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py baseline --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py baseline --max_steps 51200 --seed 92']
]

# FOCCAL (CT-Graph)
commands_ctgraph_lc = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92']
]

commands_ctgraph_sc = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    #['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92']
]

commands_minigrid = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --max_steps 409600 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask hyla --max_steps 409600 --seed 92']
]

commands_ctgraph_individual = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task1 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task1.json'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task6 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task6.json'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task17 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task17.json'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task39 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task39.json'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task82 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task82.json'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task169 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task169.json'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task342 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task342.json'],

    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4095 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4095.json'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4102 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4102.json'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4117 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4117.json'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4146 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4146.json'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4205 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4205.json'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4322 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4322.json'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4557 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4557.json'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8184 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8184.json'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8189 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8189.json'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8198 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8198.json'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8216 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8216.json'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8253 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8253.json'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8327 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8327.json'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8474 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8474.json'],

    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12279 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12279.json'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12286 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12286.json'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12301 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12301.json'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12391 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12391.json'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12510 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12510.json'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12748 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12748.json'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task13225 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task13225.json']
]

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='indicate which experiment is being run for command selection', type=str, default='ctgraph')
parser.add_argument('--exp', help='', type=str, default='')
args = parser.parse_args()
commands = None
if args.env == 'ctgraph_lc':
    commands = commands_ctgraph_lc
elif args.env == 'minigrid':
    commands = commands_minigrid
elif args.env == 'ctgraph_ppo':
    commands = commands_ctgraph_ppo
elif args.env == 'ctgraph_sc':
    commands = commands_ctgraph_sc
elif args.env == 'ctgraph_individual':
    commands = commands_ctgraph_individual
else:
    raise ValueError(f'no commands have been setup for --exp {args.exp}')


env = dict(os.environ)

path_header = args.env
if len(args.exp) > 0:
    path_header = args.exp

# Run the commands in seperate terminals
processes = []
for command in commands:
    print(f"{command[0]}, {command[1]} -p {path_header}")
    env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1] + f' -p {path_header}'), env=env)
    processes.append(process)
    #time.sleep(5)

for process in processes:
    stdout, stderr = process.communicate()