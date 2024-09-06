# HPE - POINT CLOUD

- HOH Dataset Download
- - Get the dataset from HOH Website : https://github.com/Terascale-All-sensing-Research-Studio/HOHDataset?tab=readme-ov-file
- - Put the zip dataset (e.g.01638-46157-S1_pcfiltered.zip) in HOH/pcfiltered folder
- - run the script `python3 data/HOH/extract_hoh_data.py`

- DexYCB Dataset Download
- - Download MANO models and code (mano_v1_2.zip) from the MANO website : [Mano site](https://mano.is.tue.mpg.de/) and place the file under data/DexYCB/. Unzip with:  
    `cd data/DexYCB`
    `unzip mano_v1_2.zip`
- - Download the cache [DexYCB dataset](https://dex-ycb.github.io) using `python3 data/DexYCB/dexycb_download.py`.
- - Compile the asset using `python3 data/DexYCB/compile_asset.py`
- - Run this `./data/DexYCB/compile_assets.sh` to compile the assets
- - Download calibration.tar.gz, bop.tar.gz, models.tar.gz from [DexYCB dataset](https://dex-ycb.github.io)
- - Unzip them and put them in the  `./data/DexYCB/` directory 

- Installing dependencies
- - `cd dependencies`
- - `git clone "github repo"`
- - Rename it by adding @ the beginning of the repo
- - Update the config/dependency_config.py file with the dependencies
- - Run `cd dependencies/{package_name}`
- - Run `export PYTHONPATH=$PYTHONPATH:~/Handover-imitation/dependencies/@{package_name}` # /Users/user/Documents/GitHub/Handover-imitation/
- - Run `pip install .` if the repo has setup.py to make it installable
- - Then `source ~/.bashrc`  # or source ~/.zshrc if you're using zsh

- Evaluating the dataset (DexYCB and HOH)
- - DexYCB bop and hpe evaluation
- - run `export PYTHONPATH=$PYTHONPATH:~/Documents/GitHub/Handover-imitation/data/HOH`
- - run 'export DEX_YCB_DIR^Cata/DexYCB


- Sample generated processed dataset
- - `{'subject': '7', 'frame': 'frame9173', 'giver': {'intrinsics': {'fx': 918.53385681, 'fy': 917.31624071, 'ppx': 957.75859135, 'ppy': 549.78387066}, 'hoh_ids': [7, 0, 0, 0], 'hoh_grasp_ind': 0, 'mano_side': 'right', 'mano_betas': [[-1.0759861946105957, -1.1553570747375488, -1.0234966278076172, -0.9352028846740723, 1.000255298614502, 1.0106595039367676, 0.9579319953918457, -1.0134562492370605, -0.9834078788757324, -1.1410846710205078]], 'label_file': 'data/HOH/calibration/labels/7/giver_frame9173.npz'}, 'receiver': None, 'file': 'giver_frame9173.ply'}`

- - `{'subject': '7', 'frame': 'frame9175', 'giver': None, 'receiver': {'intrinsics': {'fx': 918.53385681, 'fy': 917.31624071, 'ppx': 957.75859135, 'ppy': 549.78387066}, 'hoh_ids': [7, 0, 0, 0], 'hoh_grasp_ind': 0, 'mano_side': 'right', 'mano_betas': [[-1.0722830772399903, -1.1585710525512696, -1.0184940338134765, -0.9231698989868165, 0.9912988662719726, 1.004500389099121, 0.9512113571166992, -1.0159682273864745, -0.9841879844665528, -1.1348061561584473]], 'label_file': 'data/HOH/calibration/labels/7/receiver_frame9175.npz'}, 'file': 'receiver_frame9175.ply'}`


- Joint 3D hand pose position
- - Giver hand Pose
- - `` # Example default MANO joint_3d values for a handover pose default_joint_3d_giver = [
            [0.0, 0.0, 0.0],  # wrist
            [0.1, -0.2, 0.2],  # thumb_mcp
            [0.15, -0.25, 0.3],  # thumb_pip
            [0.2, -0.3, 0.35],  # thumb_dip
            [0.25, -0.35, 0.4],  # thumb_tip
            [0.3, 0.1, 0.5],  # index_mcp
            [0.4, 0.15, 0.55],  # index_pip
            [0.5, 0.2, 0.6],  # index_dip
            [0.6, 0.25, 0.65],  # index_tip
            [0.35, 0.0, 0.45],  # middle_mcp
            [0.45, 0.05, 0.5],  # middle_pip
            [0.55, 0.1, 0.55],  # middle_dip
            [0.65, 0.15, 0.6],  # middle_tip
            [0.4, -0.1, 0.4],  # ring_mcp
            [0.5, -0.05, 0.45],  # ring_pip
            [0.6, 0.0, 0.5],  # ring_dip
            [0.7, 0.05, 0.55],  # ring_tip
            [0.35, -0.15, 0.35],  # little_mcp
            [0.45, -0.1, 0.4],  # little_pip
            [0.55, -0.05, 0.45],  # little_dip
            [0.65, 0.0, 0.5]   # little_tip
        ]``
- - Receiver Hand Pose
- - `` # Example default MANO joint_3d values for a receiving pose
default_joint_3d_receiver = [
    [0.0, 0.0, 0.0],  # wrist
    [0.2, -0.1, 0.3],  # thumb_mcp
    [0.25, -0.15, 0.35],  # thumb_pip
    [0.3, -0.2, 0.4],  # thumb_dip
    [0.35, -0.25, 0.45],  # thumb_tip
    [0.4, 0.2, 0.5],  # index_mcp
    [0.5, 0.25, 0.55],  # index_pip
    [0.6, 0.3, 0.6],  # index_dip
    [0.7, 0.35, 0.65],  # index_tip
    [0.45, 0.1, 0.55],  # middle_mcp
    [0.55, 0.15, 0.6],  # middle_pip
    [0.65, 0.2, 0.65],  # middle_dip
    [0.75, 0.25, 0.7],  # middle_tip
    [0.5, 0.0, 0.5],  # ring_mcp
    [0.6, 0.05, 0.55],  # ring_pip
    [0.7, 0.1, 0.6],  # ring_dip
    [0.8, 0.15, 0.65],  # ring_tip
    [0.45, -0.05, 0.45],  # little_mcp
    [0.55, 0.0, 0.5],  # little_pip
    [0.65, 0.05, 0.55],  # little_dip
    [0.75, 0.1, 0.6]   # little_tip
] ``