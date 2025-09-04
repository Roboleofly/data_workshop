#!/bin/bash

# agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple
# 
# repo_ids = ['''agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple_20250714',
#             'agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_banana_20250714',
#             'agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714']

# repo_id="agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714"
# cd /root/transfer
# linux-bcecmd-0.5.1/bcecmd bos sync bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/${repo_id} /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/${repo_id}


# repo_ids=(
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet_20250721"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca_20250722"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250723"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250724"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250725"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250729"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250730"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_organize_bottom"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250804"
#     "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250805"
# )

repo_ids_1=(
    "tienkung_pro2_dualArm-gripper-1cameras_2_assemble_valve_thread_20250803"
    "tienkung_pro2_dualArm-gripper-1cameras_2_place_phillips_screwdriver_in_toolbox"
    "tienkung_pro2_dualArm-gripper-1cameras_2_pour_oil_for_gear"
    "tienkung_pro2_dualArm-gripper-1cameras_2_gather_wrench_and_screwdriver_into_toolbox"
    "tienkung_pro2_dualArm-gripper-1cameras_2_gather_wrench_and_screwdriver_into_toolbox_250801"
    "tienkung_pro2_dualArm-gripper-1cameras_2_pour_oil_for_gear_250805"
    "tienkung_pro2_dualArm-gripper-1cameras_2_close_distribution_box"
    "tienkung_pro2_dualArm-gripper-1cameras_2_place_tea_drinks_in_basket"
    "tienkung_pro2_dualArm-gripper-1cameras_2_press_stop_button_of_control_box"
    "tienkung_pro2_dualArm-gripper-1cameras_2_close_distribution_box_250806"
    "tienkung_pro2_dualArm-gripper-1cameras_2_close_distribution_box_250808"
    "tienkung_pro2_dualArm-gripper-1cameras_2_press_stop_button_of_control_box_250812"
    "tienkung_pro2_dualArm-gripper-1cameras_2_1_find_out_circuit_breaker_into_the_other_tray"
    "tienkung_pro2_dualArm-gripper-1cameras_2_find_out_red_capacitance_into_the_other_tray_copy_1755064314752"
    "tienkung_pro2_dualArm-gripper-1cameras_2_find_out_packaging_tape_into_the_other_box_250813"
    "tienkung_pro2_2_find_out_packaging_tape_into_the_other_basket_250814"
)


repo_ids_2=(
    "tienkung_pro2_2_weight_sauce" #berfore                               
    "pick_up_parts_from_belt_conveyor_place_on_plate_fast_250722" #before
    "pick_up_parts_from_belt_conveyor_place_on_plate_fast_250721" #before
    "tienkung_pro2_2_place_the_milk_250818"
    "tienkung_pro2_2_weight_sauce_250819"
    "tienkung_pro2_dualArm-gripper-1cameras_2_place_biscuit_box_on_shelf"
    "tienkung_pro2_dualArm-gripper-1cameras_2_gather_long_short_screws_in_storage_box"
    "tienkung_pro2_dualArm-gripper-1cameras_2_gather_long_short_screws_in_storage_box_250820"
)


repo_ids_3=(
    "tienkung_pro2_2_weight_sauce" #berfore                               
)


cd /root/transfer

# for repo_id in "${repo_ids[@]}"; do
#     echo "正在下载: $repo_id"
#     linux-bcecmd-0.5.1/bcecmd bos sync \
#         bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/${repo_id} \
#         /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/${repo_id}
# done


for repo_id in "${repo_ids_2[@]}"; do
    echo "正在下载: $repo_id"
    linux-bcecmd-0.5.1/bcecmd bos sync \
        bos:/data-collections-bd/raw_data/tienkung_pro2_dualArm-gripper-1cameras_2/${repo_id} \
        /media/jushen/gongda-wu/research/datasets/station_data/pkl_data/tienkung_pro2_dualArm-gripper-1cameras_2/${repo_id}
done

# cd /root/transfer
# linux-bcecmd-0.5.1/bcecmd bos sync bos:/data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714/ /media/jushen/leofly-liao/datasets/pickle/agilex/mobile/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714/


# data-collections-bd/raw_data/agilex_cobotmagic2_dualArm-gripper-3cameras_5/agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn

# agilex_cobotmagic3_dualArm-gripper-3cameras_2_find_out_packaging_tape_into_the_other_basket_20250703