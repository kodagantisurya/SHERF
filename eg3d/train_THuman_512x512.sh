python -u train.py --outdir=logs/training-THuman-runs-90-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4 --cfg=THuman --data=/mnt/lustre/skhu/eg3d/eg3d-v1/data/THuman/nerf_data_/results_gyc_20181010_hsc_1_M --gpus=4 --batch=4 --gamma=5 --aug=noaug --neural_rendering_resolution_initial=512 --gen_pose_cond=True --gpc_reg_prob=0.8 --kimg 800 --workers 3 --use_1d_feature True --use_2d_feature True --use_3d_feature True --use_sr_module False --sample_obs_view True --fix_obs_view False --use_nerf_decoder True --use_trans True