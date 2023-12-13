GROUP-9(SHERF)

Introduction:
Implementation details:
•	There are two ways we can try to execute this code.
i ) Local Environment
ii ) Online platforms like Google Colab or GCP where we can have additional GPUs for execution.
•	We need to install all necessary packages that can be downloaded by running the requirement.txt in either one of the environments.
•	We require 4GPUs for executing this code without facing any errors.
•	When we have less GPUs we can modify the bash code which looks like eval_renderpeople.sh where can set number of GPUs to 2 or 1 based on our resources.
•	We need to install pretrained SMPL models called SMPL_neutral.pkl and SHERF_renderpeople.pkl. These models were already trained and will help in reducing the complexity of our execution.
•	As mentioned in the Github repository all the datasets access should be obtained and downloaded and should extract that dataset in data folder.
•	The two pretrained SMPL models should be placed in two different directories. The SMPL_NEUTRAL.pkl should be placed in new folder called assets that should be created in sherf and SHERF_renderpeople.pkl or other model based on dataset should be placed in models->smpl->place it here.
•	Datasets:
Except Renderpeople dataset all other datasets need access and also need huge storage to save and extract them in the directory.
•	Executing in LOCAL environment:
o	This program requires distributed processing for execution.
o	So, we need to install Ubuntu and implement WSL in command prompt for this execution.
o	We need to install NVIDIA CUDA drivers (toolkit 11 or above version).
o	Windows Subsystem for Linux (WSL) helps in dividing our process to implement the process simultaneously.
o	When we have enough GPUs at first, we can try to inference our code with command as this, bash eval_renderpeople_512x512.sh.
o	When we want to train the model, we need to execute our code in command prompt by giving this command, bash train_renderpeople_512x512.sh
o	We might face issues like dimensions mismatch or cuda is not set when we don’t have Nvidia drivers installed or not sufficient GPUs in the PC.
o	If we don’t have enough GPUs we can try to execute the code by modifying the code as below, python -u train.py --outdir=logs/training-RenderPeople-runs-450-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4 --cfg=RenderPeople --data=data/RenderPeople_recon/20230228/seq_000000-rp_aaron_rigged_001 --gpus=1 --batch=4 --gamma=5 --aug=noaug --neural_rendering_resolution_initial=512 --gen_pose_cond=True --gpc_reg_prob=0.8 --kimg 800 --workers 3 --use_1d_feature True --use_2d_feature True --use_3d_feature True --use_sr_module False --sample_obs_view True --fix_obs_view False --use_nerf_decoder True --use_trans True --test_flag True --resume logs/training-RenderPeople-runs-450-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4/SHERF_RenderPeople.pkl
o	If we don’t have GPUs and no cuda drivers then we will not be able to execute the code in the local environment.

•	GOOGLE COLAB:
o	The google colab by default provides us one GPU for the code execution.
o	So, we need to clone our SHERF Github repository and datasets, pretrained models should be placed in the directory as mentioned above.
o	We need to install all packages as given in my notebook.
o	When we tried to execute the commands as given in GPU-1 code then we will get error in smpl_numpy.py file where we will have some dimensions mismatch.
o	We need to reshape the arrays to match all tensor arrays.
o	But still there has been more errors in dimensions in volume_rendering.py
o	When we will try to fix all those dimension errors the code will work fine and produces our required output.
•	Challenges:
o	We don’t have as many GPUs required so we got struck with lot of errors like context has already been set and others.
o	When we ran the code in local system with GPU set to one, we got error struck with dimension mismatch between shapedirs and v_template to perform torch operations.
o	So we tried to execute the code in Google colab which has GPU built with in it and when we tried to execute that code we again caught with same errors.
o	When we tried to resolve the issue we got an error  saying that CUDA and GPU in colab (tensors t4) were not properly aligned i.e., needs to be run in same environment.
o	So we need to 4GPUs to exactly implement this code.Even for inferencing the model it requires more number of GPUs as there are lot of dimesniosn which are mismatched due to unavailability of resources like number of GPU,different CUDA / Nvidia drivers.		

