#!/bin/bash

# agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple
# 
# repo_ids = ['''agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple_20250714',
#             'agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_banana_20250714',
#             'agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714']

# repo_id="agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714"
# cd /root/transfer
# linux-bcecmd-0.5.1/bcecmd bos sync bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/${repo_id} /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/${repo_id}


repo_ids=(
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet_20250721"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca_20250722"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250723"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250724"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250725"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250729"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250730"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_organize_bottom"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250804"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250805"
)

cd /root/transfer

for repo_id in "${repo_ids[@]}"; do
    echo "正在下载: $repo_id"
    linux-bcecmd-0.5.1/bcecmd bos sync \
        bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/${repo_id} \
        /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/${repo_id}
done

# cd /root/transfer
# linux-bcecmd-0.5.1/bcecmd bos sync bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714/ /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714/


# data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn

# agilex_cobotmagic3_dualArm-gripper-3cameras_2_find_out_packaging_tape_into_the_other_basket_20250703