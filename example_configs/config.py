first_inv_type = 'w+_cam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
lpips_type = 'alex'
pt_l2_lambda = 1
pt_lpips_lambda = 1
landmark = 0.5                 # 0.5 or 1
mask_border = True

## Steps
first_inv_steps = 1000         # 500 ~ 1000
camera_opt_step = 300          # 150 ~ 300
LPIPS_value_threshold = 0.06   # where to stop the optimization
max_pti_steps = 500            # max number of steps for the pti
max_images_to_invert = 150     # 

## Optimization
optim_type = 'adam'
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
first_inv_cam_lr = 1e-2
rotation_lr_scale = 0.1        # lr scale for the rotation term
lambda_lr_scale = 0.1          # lr scale for the approximation term
lmk_lr_scale = 0.1             # lr scale for the uncertainty term
train_batch_size = 1
use_last_w_pivots = True       # True for loading the pivots from the last inversion
norm_init = 0.3                # initial value for distance
cam_opt_random_ws = True
only_generator = True

## Inversion identifier
method = 'disco'
input_data_id = 'wild'
pti_results_keyword = 'v1-random-init0.3'

## Paths
unprocessed_folder = '/path/to/unprocessed/images'       ## where the unprocessed image
input_data_path = f'../../data/{input_data_id}/crop'  ## where the processed image
eg3d_dir = '/net/per610a/export/das18a/satoh-lab/wangzx/src/Perspective/eg3d'
deep3d_path = f'{eg3d_dir}/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch'
# Pretrained models paths
stylegan2_ada_ffhq = 'pretrained_models/ffhq512-128.pkl'

## Dirs for output files
out_folder = 'exp'
run_name = f'{method}-{input_data_id}-{pti_results_keyword}'
base_folder = f'../../{out_folder}/{run_name}'
checkpoints_dir = f'{base_folder}/checkpoints'
embedding_base_dir = f'{base_folder}/embeddings'
experiments_output_dir = f'{base_folder}/output'
wandb_dir = base_folder
debug = True
debug_dir = f'{base_folder}/debug'