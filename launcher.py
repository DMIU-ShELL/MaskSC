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

commands_ctgraph_oracle = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --max_steps 51200 --seed 92']
]

commands_ctgraph_sc_random = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 86'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 87'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 88'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 89'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 90'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 91'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    #['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 86'],
    #['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 87'],
    #['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 88'],
    #['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 89'],
    #['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 90'],
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 91'],
    #['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy random_topk --max_steps 51200 --seed 92']
]

commands_ctgraph_single_task_experts = [
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

    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12278 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12278.json'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12284 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12284.json'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12297 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12297.json'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12323 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12323.json'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12375 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12375.json'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12478 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12478.json'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12684 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12684.json']
]


commands_minigrid = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input'],

    # MIG 2 (7-13)
    #['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    #['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    #['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    #['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    #['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    #['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_oracle = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_all --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input'],

    # MIG 2 (7-13)
    #['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    #['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    #['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    #['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    #['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    #['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input']
]

MIG_1_IDS = [
    'MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277',
    'MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa',
    'MIG-c432df19-0894-5232-ac1c-9a3440fc267e',
    'MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf',
    'MIG-35ecef79-db2e-590b-9e8c-2c07c787008e',
    'MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0',
    'MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e',
]

MIG_2_IDS = [
    'MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3',
    'MIG-4590f80d-be70-58e4-af75-eeb950255d4a',
    'MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2',
    'MIG-2593b912-5975-58e9-bc3d-495311cee807',
    'MIG-51069529-f343-59c6-bac7-a75648296e7b',
    'MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7',
    'MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1',
]

MIG_GROUPS = {
    0: MIG_1_IDS,
    1: MIG_2_IDS,
}


def make_minigrid_single_task_expert_commands():
    commands = []
    config_seed_by_run_seed = {
        86: 841,
        87: 845,
        88: 849,
        89: 853,
        90: 857,
        91: 861,
        92: 865,
    }

    for run_seed, config_seed in config_seed_by_run_seed.items():
        # One 16-command block per RL seed, matching env_configs/minigrid_object_remap_seed{run_seed}.json.
        for task_idx in range(16):
            mig_id = MIG_1_IDS[task_idx % len(MIG_1_IDS)]
            env_config_path = (
                f'./env_configs/mg16_remap/seed{run_seed}/'
                f'minigrid_object_remap_seed{config_seed}_{task_idx}.json'
            )
            command = (
                'python train_minigrid.py baseline '
                f'--env_config_path {env_config_path} '
                f'--exp_id task{task_idx} '
                '--max_steps 512_000 '
                f'--seed {run_seed} '
                '--disable_task_label_input'
            )
            commands.append([mig_id, command, f'seed{run_seed}'])
    return commands


commands_minigrid_single_task_experts = make_minigrid_single_task_expert_commands()

commands_continualworld = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 92 --disable_task_label_input']
]

COMMAND_SETS = {
    'ctgraph_lc': commands_ctgraph_lc,
    'minigrid': commands_minigrid,
    'minigrid_individual': commands_minigrid_single_task_experts,
    'minigrid_oracle': commands_minigrid_oracle,
    'ctgraph_ppo': commands_ctgraph_ppo,
    'ctgraph_sc': commands_ctgraph_sc,
    'ctgraph_oracle': commands_ctgraph_oracle,
    'ctgraph_individual': commands_ctgraph_single_task_experts,
    'ctgraph_sc_random': commands_ctgraph_sc_random,
    'continualworld': commands_continualworld,
}


def parse_gpu_groups(raw_values):
    if raw_values is None:
        return None

    # Accept: --gpu 0 1, --gpu 0,1, --gpu "[0, 1]", or --gpu [0, 1].
    cleaned = ' '.join(raw_values).replace('[', ' ').replace(']', ' ').replace(',', ' ')
    values = cleaned.split()
    if not values:
        raise ValueError('--gpu requires at least one MIG group')

    groups = []
    for value in values:
        try:
            group = int(value)
        except ValueError as exc:
            raise ValueError(f'invalid MIG group {value!r}; available groups: {sorted(MIG_GROUPS)}') from exc
        if group not in MIG_GROUPS:
            raise ValueError(f'unknown MIG group {group}; available groups: {sorted(MIG_GROUPS)}')
        if group not in groups:
            groups.append(group)
    return groups


def build_mig_pool(group_ids):
    """Interleave selected groups so multi-group runs use every physical GPU."""
    pool = []
    max_group_size = max(len(MIG_GROUPS[group]) for group in group_ids)
    for slot in range(max_group_size):
        for group in group_ids:
            if slot < len(MIG_GROUPS[group]):
                pool.append(MIG_GROUPS[group][slot])
    return pool


def assign_mig_ids(commands, group_ids):
    if group_ids is None:
        return [command[0] for command in commands]

    pool = build_mig_pool(group_ids)
    return [pool[index % len(pool)] for index in range(len(commands))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        help='indicate which experiment is being run for command selection',
        type=str,
        default='ctgraph_sc',
        choices=sorted(COMMAND_SETS),
    )
    parser.add_argument('--exp', help='', type=str, default='')
    parser.add_argument(
        '--legacy_wte_ema',
        '--legacy-wte-ema',
        dest='legacy_wte_ema',
        help='append the legacy WTE EMA compatibility flag to every command',
        action='store_true',
    )
    parser.add_argument(
        '--gpu',
        nargs='+',
        metavar='GROUP',
        help=(
            'MIG group(s) used to schedule commands: 0 selects the first seven '
            'MIG instances and 1 selects the second seven. Multiple groups are '
            'interleaved. Examples: --gpu 0, --gpu 0 1, --gpu "[0, 1]". '
            'If omitted, each command keeps its hard-coded MIG UUID.'
        ),
    )
    args = parser.parse_args()

    commands = COMMAND_SETS[args.env]
    gpu_groups = parse_gpu_groups(args.gpu)
    assigned_mig_ids = assign_mig_ids(commands, gpu_groups)

    path_header = args.exp if args.exp else args.env
    
    processes = []
    for command, mig_id in zip(commands, assigned_mig_ids):
        command_path_header = path_header
        if len(command) > 2 and command[2]:
            command_path_header = os.path.join(path_header, command[2])

        command_text = command[1] + f' -p {command_path_header}'
        if args.legacy_wte_ema:
            command_text += ' --legacy_wte_ema'
        print(f'{mig_id}, {command_text}')

        process_env = dict(os.environ)
        process_env['CUDA_VISIBLE_DEVICES'] = mig_id
        process = subprocess.Popen(shlex.split(command_text), env=process_env)
        processes.append(process)
        # time.sleep(5)

    for process in processes:
        process.communicate()


if __name__ == '__main__':
    main()
