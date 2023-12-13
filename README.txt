                                                  GROUP-9 
-----------------------------------------------------------------------------------------------------------------------------
                            SHERF - Generalizable Human NeRF from a Single Image
-----------------------------------------------------------------------------------------------------------------------------
Introduction:
Introducing SHERF, the revolutionary 3D human model that breathes life into a single image! SHERF unlocks the potential to render and animate realistic humans from just one snapshot, overcoming the limitations of traditional methods. It does this by extracting and encoding the essence of 3D human form in a universal space, allowing for animation and rendering from any angle or pose. This is achieved through a unique combination of 3D-aware features, capturing global, point-level, and pixel-aligned details, creating an incredibly informative representation. Extensive tests on various datasets prove SHERF's superior performance, pushing the boundaries of what's possible with just a single image.
-----------------------------------------------------------------------------------------------------------------------------
SYSTEM REQUIREMENTS:
1) Windows Subsystem for LINUX(UBUNTU).
2) 4 GPUs required for execution.
3) 100GB hard drive storage for datasets.
4) Minimum 8gb RAM
5) NVIDIA GPU with all ncessary drivers and CUDA toolkit
6) Anaconda and python(min 3.8) environment setup.
-----------------------------------------------------------------------------------------------------------------------------
Datasets:(Need access except RenderPeople)
1)RenderPeople - https://mycuhk-my.sharepoint.com/:f:/g/personal/1155098117_link_cuhk_edu_hk/ElL9IDDOaa5Hl785gvbqyEEB8ubdobyuMKqoDY3J85XStw?e=o2BUOt
2)T-HUMAN - https://github.com/gaoxiangjun/MPS-NeRF
3)HuMMan - https://caizhongang.github.io/projects/HuMMan/
4)ZU-Mocap - https://github.com/zju3dv/neuralbody
-----------------------------------------------------------------------------------------------------------------------------
Pre-trained models:
1)SHERF_renderpeople or other datasets pretrained model:
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155098117_link_cuhk_edu_hk/EU3RxpLuKmZImkdJbG8Y12EBZ9RxIfQiEx7ctt5obXUjzw?e=gXJbIQ
2) SMPL NEUTRAL model - https://smpl.is.tue.mpg.de/
-----------------------------------------------------------------------------------------------------------------------------
Implementation details:
-> There are two ways we can try to execute this code.
i ) Local Environment
ii ) Online platforms like Google Colab or GCP where we can have additional GPUs for execution.
-> We need to install all necessary packages that can be downloaded by running the requirement.txt in either one of the environments.
-> We require 4GPUs for executing this code without facing any errors.
-> When we have less GPUs we can modify the bash code which looks like eval_renderpeople.sh where can set number of GPUs to 2 or 1 based on our resources.
-> We need to install pretrained SMPL models called SMPL_neutral.pkl and SHERF_renderpeople.pkl. These models were already trained and will help in reducing the complexity of our execution.
-> As mentioned in the Github repository all the datasets access should be obtained and downloaded and should extract that dataset in data folder.
-> The two pretrained SMPL models should be placed in two different directories. The SMPL_NEUTRAL.pkl should be placed in new folder called assets that should be created in sherf and SHERF_renderpeople.pkl or other model based on dataset should be placed in models->smpl->place it here.

Executing in LOCAL environment:
-> This program requires distributed processing for execution.
-> So, we need to install Ubuntu and implement WSL in command prompt for this execution.
-> We need to install NVIDIA CUDA drivers (toolkit 11 or above version).
->	Windows Subsystem for Linux (WSL) helps in dividing our process to implement the process simultaneously.
->	When we have enough GPUs at first, we can try to inference our code with command as this, bash eval_renderpeople_512x512.sh.
->	When we want to train the model, we need to execute our code in command prompt by giving this command, bash train_renderpeople_512x512.sh
->	We might face issues like dimensions mismatch or cuda is not set when we don’t have Nvidia drivers installed or not sufficient GPUs in the PC.
->	If we don’t have enough GPUs we can try to execute the code by modifying the code as below,
python -u train.py --outdir=logs/training-RenderPeople-runs-450-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4 --cfg=RenderPeople --data=data/RenderPeople_recon/20230228/seq_000000-rp_aaron_rigged_001 --gpus=1 --batch=4 --gamma=5 --aug=noaug --neural_rendering_resolution_initial=512 --gen_pose_cond=True --gpc_reg_prob=0.8 --kimg 800 --workers 3 --use_1d_feature True --use_2d_feature True --use_3d_feature True --use_sr_module False --sample_obs_view True --fix_obs_view False --use_nerf_decoder True --use_trans True --test_flag True --resume logs/training-RenderPeople-runs-450-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4/SHERF_RenderPeople.pkl
->	If we don’t have GPUs and no cuda drivers then we will not be able to execute the code in the local environment.

GOOGLE COLAB:
->	The google colab by default provides us one GPU for the code execution.
->	So, we need to clone our SHERF Github repository and datasets, pretrained models should be placed in the directory as mentioned above.
->	We need to install all packages as given in my notebook.
->	When we tried to execute the commands as given in GPU-1 code then we will get error in smpl_numpy.py file where we will have some dimensions mismatch.
->	We need to reshape the arrays to match all tensor arrays.
->	But still there has been more errors in dimensions in volume_rendering.py
->	When we will try to fix all those dimension errors the code will work fine and produces our required output.
-----------------------------------------------------------------------------------------------------------------------------
Challenges:
->	We don’t have as many GPUs required so we got struck with lot of errors like context has already been set and others.
->	When we ran the code in local system with GPU set to one, we got error struck with dimension mismatch between shapedirs and v_template to perform torch operations.
->	So we tried to execute the code in Google colab which has GPU built with in it and when we tried to execute that code we again caught with same errors.
->	When we tried to resolve the issue we got an error  saying that CUDA and GPU in colab (tensors t4) were not properly aligned i.e., needs to be run in same environment.
->	So we need to 4GPUs to exactly implement this code.Even for inferencing the model it requires more number of GPUs as there are lot of dimesniosn which are mismatched due to unavailability of resources like number of GPU,different CUDA / Nvidia drivers.		

