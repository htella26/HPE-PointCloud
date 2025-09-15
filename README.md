
Hand Pose Estimation (HPE) in Point Cloud
Hand Pose Estimation

![Hand Pose Estimation](assets/Subjects_images.png)

Overview
This repository contains code and data preparation scripts for estimating 3D hand poses from point cloud representations of handover interactions. We focus on the Human‑Object‑Human (HOH) dataset and the DexYCB dataset, providing a processing pipeline, evaluation tools and visualization scripts. The aim is to facilitate research on hand pose estimation for robotic grasping and human–robot handover tasks.
Key features:
•	Extraction of hand and object point clouds from multi‑modal HOH recordings.
•	Preparation of DexYCB dataset including MANO hand models and object meshes.
•	Pre‑trained models and evaluation scripts for hand pose estimation (HPE) on HOH and DexYCB.
•	Scripts to visualise point clouds and 3D hand joint predictions.
 
Table of Contents
1.	Dataset Setup
2.	Dependencies and Installation
3.	Data Preparation
4.	Running Evaluations
5.	Visualization
6.	Sample Data Format
7.	Citation
8.	License
9.	Contact
 
Dataset Setup
HOH Dataset
The HOH (Human‑Object‑Human) dataset contains multi‑view RGB and depth data of two‑person handover interactions[1]. To use this dataset:
1.	Download the HOH dataset from the official repository: https://github.com/Terascale-All-sensing-Research-Studio/HOHDataset?tab=readme-ov-file.
2.	Place each downloaded pcfiltered zip file (for example 01638-46157-S1_pcfiltered.zip) into data/HOH/pcfiltered/.
3.	Extract the point clouds into a usable format by running the following script:
python3 data/HOH/extract_hoh_data.py
This script will unpack the point clouds and calibration files into data/HOH/filtered/ and organise the handover sessions into train/validation/test splits.
DexYCB Dataset
For the DexYCB dataset you need the MANO hand model and the object meshes:
1.	Download the MANO hand model archive (mano_v1_2.zip) from the MANO website and place it under data/DexYCB/. Unzip it:
cd data/DexYCB
unzip mano_v1_2.zip
1.	Download the DexYCB dataset using the provided script (this requires a few GB of disk space):
python3 data/DexYCB/dexycb_download.py
1.	Compile the dataset assets (meshes, camera parameters, etc.):
python3 data/DexYCB/compile_asset.py
# or run the helper script
./data/DexYCB/compile_assets.sh
1.	From the DexYCB website, download the archives calibration.tar.gz, bop.tar.gz and models.tar.gz, then extract them into the data/DexYCB/ directory.
Additional Data
A small preprocessed sample dataset is provided for quick experimentation. Download data.zip from the following Google Drive link:
https://drive.google.com/file/d/151iVtflMms-icXmLNNAN8gP10rgjsywh/view?usp=share_link
After downloading, unzip the archive into the repository root so that the contents appear under data/.
 
Dependencies and Installation
This repository uses Python 3.8+. Install the base Python dependencies with:
pip install -r requirements.txt
Some evaluation scripts depend on external toolkits which must be cloned into the dependencies/ folder. You can either run the helper script or clone them manually:
1.	Using the helper script (requires Google Drive access)
python3 get_dependency.py
This will download and extract a pre‑packaged set of dependencies into dependencies/.
1.	Manual installation
Clone each dependency into the dependencies/ folder and rename the directory to include a @ prefix:
cd dependencies
git clone https://github.com/thodan/bop_toolkit.git           # rename to @bop_toolkit
git clone https://github.com/versatran01/dex-ycb-toolkit.git  # rename to @dex-ycb-toolkit
git clone https://github.com/eth-siplab/freihand.git          # rename to @freihand
git clone https://github.com/hassony2/manopth.git             # rename to @manopth
After cloning, update the paths in config/dependency_config.py if necessary and add each package to your PYTHONPATH:
export PYTHONPATH=$PYTHONPATH:/path/to/HPE-PointCloud/dependencies/@bop_toolkit
export PYTHONPATH=$PYTHONPATH:/path/to/HPE-PointCloud/dependencies/@dex-ycb-toolkit
export PYTHONPATH=$PYTHONPATH:/path/to/HPE-PointCloud/dependencies/@freihand
export PYTHONPATH=$PYTHONPATH:/path/to/HPE-PointCloud/dependencies/@manopth
On Linux/macOS you can add these lines to your .bashrc or .zshrc to set them automatically.
 
Data Preparation
Once the datasets and dependencies are in place, prepare the HOH point clouds by running:
python3 data/HOH/extract_hoh_data.py
This will create a structured directory under data/HOH/filtered/ containing the giver hand, receiver hand and object point clouds alongside the camera intrinsics and MANO parameters.
For DexYCB, the download and compilation scripts described above will prepare the dataset for evaluation.
 
Running Evaluations
We provide evaluation scripts for both datasets. These scripts assume that the environment variables DEX_YCB_DIR and HOH_ROOT point to the DexYCB and HOH data directories respectively.
DexYCB Evaluation
To evaluate a predicted results file on DexYCB, run:
export DEX_YCB_DIR=/path/to/data/DexYCB    # adjust this path
python3 evaluation/dexycb_eval.py \
    --eval_type hpe \
    --res_file evaluation/dexycb_sample_results/example_results_hpe_s0_test.txt \
    --out_dir evaluation/dexycb_sample_results/ \
    --visualize
The evaluator will compute standard metrics and optionally visualise the joint predictions.
HOH Evaluation
For HOH, the evaluation differs by hand type (giver or receiver). Provide the appropriate results file:
export HOH_ROOT=/path/to/data/HOH
# Evaluate the giver hand
python3 evaluation/hoh_eval.py \
    --hand_type giver \
    --res_giver_file evaluation/hoh_eval_results/giver_hpe_predicted_results_test.txt

# Evaluate the receiver hand
python3 evaluation/hoh_eval.py \
    --hand_type receiver \
    --res_receiver_file evaluation/hoh_eval_results/receiver_hpe_predicted_results_test.txt
The evaluator will output joint error metrics and optionally visualise the predicted poses.
 
Visualization
Several scripts are provided to visualise point clouds and hand poses:
•	visualization/pc_visualization.py — combines multiple .ply files (giver hand, object and receiver hand) and displays them using Open3D.
•	visualization/giver_hand_pose.py — plots the default MANO joint positions for a giver hand in 3D using Matplotlib.
•	visualization/giver_reciever.py — overlays giver and receiver poses to compare hand configurations.
These scripts are stand‑alone and can be run directly once the dependencies (open3d, matplotlib, numpy) are installed.
 
Sample Data Format
Processed HOH frames are stored as Python dictionaries in .npz files. Each dictionary contains metadata and MANO parameters for either the giver or receiver. An example entry looks like:
{
    'subject': '7',
    'frame': 'frame9173',
    'giver': {
        'intrinsics': {'fx': 918.53, 'fy': 917.32, 'ppx': 957.76, 'ppy': 549.78},
        'hoh_ids': [7, 0, 0, 0],
        'hoh_grasp_ind': 0,
        'mano_side': 'right',
        'mano_betas': [[-1.07, -1.16, -1.02, -0.94, 1.00, 1.01, 0.96, -1.01, -0.98, -1.14]],
        'label_file': 'data/HOH/calibration/labels/7/giver_frame9173.npz'
    },
    'receiver': None,
    'file': 'giver_frame9173.ply'
}
Similar structures exist for receiver frames with the roles reversed. Refer to the data/HOH directory after running the extraction script for more examples.
For testing and visualisation, we define default MANO joint coordinates in the visualization scripts. See the commented arrays in this README or in visualization/constant.py for sample coordinate values.
 
Citation
If you use this code or the HOH dataset in your research, please cite the following:
•	HOH Dataset: Noah Wiederhold, et al. HOH: Markerless Multimodal Human–Object–Human Handover Dataset with Large Object Count. arXiv:2310.00723 [cs.CV], 2023[1].
•	DexYCB Dataset: Yufei Yang, et al. DexYCB: A benchmark for capturing hand–object interaction.
•	MANO Model: Javier Romero, Dimitrios Tzionas and Michael J. Black. Embodied Hands: Modeling and Capturing Hands and Bodies Together.
 
License
This project is licensed under the MIT License – see the LICENSE file for details.
The HOH and DexYCB datasets may have additional restrictions; consult their respective repositories for terms of use.
 
 
[1] GitHub - htella26/HPE-PointCloud
https://github.com/htella26/HPE-PointCloud







# Hand Pose Estimation (HPE) in Point Cloud
- - The README is not yet completed (Needs More Work)
![Hand Pose Estimation](assets/Subjects_images.png)

- HOH Dataset Download
- - Get the dataset from HOH Website : https://github.com/Terascale-All-sensing-Research-Studio/HOHDataset?tab=readme-ov-file
- - Put the zip dataset (e.g.01638-46157-S1_pcfiltered.zip) in HOH/pcfiltered folder
- - run the script `python3 data/HOH/extract_hoh_data.py`

- Dataset Download and Dependencies
- - Download from the Google Drive using the link: https://drive.google.com/file/d/151iVtflMms-icXmLNNAN8gP10rgjsywh/view?usp=share_link
- - Put the unzip data (e.g.data.zip) in HPE-POINTCLOUD root directory as HPE-POINTCLOUD/data

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
