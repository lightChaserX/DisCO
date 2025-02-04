import copy
from random import sample
import wandb
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dlib
from configs import global_config
import dnnlib
from utils.log_utils import log_images_from_w, to_img
from utils.eval_utils import MN, extract3d_landmark, get_eyes
from camera_utils import LookAtPoseSampler, make_c, rotation2orth, ortho2rotation
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
max_numoffaces = 3
bool_notracking = True


def get_landmark(img, dim=2):
    """get landmark with mediapipe
    :return: np.array shape=(68, 2)
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=bool_notracking,
            max_num_faces=max_numoffaces,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        results = face_mesh.process(img)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        pts3D = np.zeros((468, 3))
        for i in range(468):
            pts3D[i, 0] = face_landmarks.landmark[i].x
            pts3D[i, 1] = face_landmarks.landmark[i].y
            pts3D[i, 2] = face_landmarks.landmark[i].z

    if dim == 2:
        lm = pts3D[:, :2]
    elif dim ==3:
        lm = pts3D
    else:
        raise NotImplementedError
    return lm

def lm_loss(pred_lms, gt_lms, weight=None):
    if weight is not None:
        sigmma2 = weight.reshape(1, -1) ** 2
        lambda_lmd = get_lm_weights(weight.reshape(1, -1))
        log_sigmma = torch.log(sigmma2)
        loss = torch.sum(torch.square(pred_lms - gt_lms), dim=2) / (sigmma2)
        loss = loss + log_sigmma
    else:
        lambda_lmd = 1
        loss = torch.sum(torch.square(pred_lms - gt_lms), dim=2)
    loss = torch.mean((lambda_lmd*loss).sum(-1))
    return loss, lambda_lmd[0, :] / sigmma2[0, :]

## todo: give weight to nose
def get_lm_weights(weight):
    w = torch.ones_like(weight)
    nose_idx = [8, 193, 168, 417, 245, 122, 6, 351, 465, 188, 196, 197, 419, 412, 174, 3, 195, 248, 399, 217, 236, 51,
           5, 281, 456, 437, 198, 134, 363, 420, 131, 220, 45, 4, 275, 440, 360, 114, 344, 240, 219, 218, 237, 44,
           125, 274, 457, 438, 439, 460, 97, 326, 19, 1, 2, 98, 327]
    w[0, nose_idx] = 1

    norm_w = w / w.sum()
    return norm_w


def project(
        G,
        target: torch.Tensor,  # [C,H,W]; Dynamic range here is [0,255], W & H must match G output resolution
        pose: torch.Tensor,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        hyperparameters=None,
        border=None,
        mask_non_face=None
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), "target shape is not correct"

    truncation_psi = 1
    truncation_cutoff = 14
    def logprint(*args):
        if verbose:
            print(*args)


    norm_init = hyperparameters.norm_init if hasattr(hyperparameters, 'norm_init') else None
    use_lmd = True if hasattr(hyperparameters, 'landmark') and hyperparameters.landmark > 0 else False
    lmd_weight = hyperparameters.landmark if use_lmd else 0

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    #--------------------------------------------------
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=2.7, device=device)

    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), pose[:,16:]], 1)
    conditioning_params = conditioning_params.expand([w_avg_samples, conditioning_params.shape[1]])
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), conditioning_params,
                          truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    #--------------------------------------------------
    # Landmarks and features for target image.
    lm_target_np = get_landmark((target.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8))
    if lm_target_np is None:
        print("Do not find a face, will not use landmark!")
        use_lmd = False
    else:
        lm_target = torch.from_numpy(lm_target_np).to(device)
        lmd_uncertainty =  torch.nn.Parameter(torch.ones(468).to(device).requires_grad_(True))

    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    ratio = 1
    if target_images.shape[2] > 256:
        original_size = target_images.shape[2]
        ratio = 256 / original_size
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')

    # Crop border or non-face regions
    if border is not None:
        upper_b = int(border['upper'] * ratio)
        down_b  = int(border['down']  * ratio)
        left_b  = int(border['left']  * ratio)
        right_b = int(border['right'] * ratio)
        target_images = target_images[:, :, upper_b:down_b, left_b:right_b]
    elif mask_non_face is not None:
        if ratio != 1:
            mask_non_face = F.interpolate(mask_non_face, size=(256, 256), mode='area')
        target_images = target_images * mask_non_face

    target_features = vgg16(target_images, resize_images=False, return_lpips=True)


    start_w = np.repeat(start_w, G.backbone.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True) 


    print(f'Initialize pose with: {pose}.')
    pose.requires_grad = False

    extrinsic = pose.squeeze()[:16].reshape(4, 4).clone()
    intrinsic = pose.squeeze()[16:]
    focal_length_init = intrinsic[0]
    ortho = rotation2orth(extrinsic[:3, :3]) # use orthogonal rotation


    rendered_mean_face = G.synthesis(w_opt, pose, noise_mode='const', force_fp32=True)
    with mp_face_mesh.FaceMesh(
            static_image_mode=bool_notracking,
            max_num_faces=max_numoffaces,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:
        init_img = to_img(rendered_mean_face)
        pts3D, _ = extract3d_landmark(init_img, face_mesh)
    pts3D_rescale = pts3D[:,:2].copy()
    lm_eye_right, lm_eye_left, lm_mouth_outer = get_eyes(pts3D_rescale)
    lmk_for_dist = np.concatenate((np.array([[(lm_eye_right[i, 0]), (lm_eye_right[i, 1])] for i in range(lm_eye_right.shape[0])]),
                                   np.array([[(lm_eye_left[i, 0]), (lm_eye_left[i, 1])] for i in range(lm_eye_left.shape[0])]),
                                   np.array([[(lm_mouth_outer[i, 0]), (lm_mouth_outer[i, 1])] for i in range(lm_mouth_outer.shape[0])])),
                                  axis=0)
    d_0 = (rendered_mean_face["image_depth"].detach().cpu().numpy()[0, 0, (128 * lmk_for_dist[:, 1]).astype(np.int32),
                                                                  (128 * lmk_for_dist[:, 0]).astype(np.int32)]).mean()
    
    assert norm_init is not None, "norm_init should be indicated"

    lm_synthz_0 = torch.from_numpy(pts3D)
    # get the depth for each landmark
    depth_0 = rendered_mean_face["image_depth"].detach()[0, 0, (lm_synthz_0[:, 1] * 128).to(torch.long), 
                                                               (lm_synthz_0[:, 0] * 128).to(torch.long)]

    imx = to_img(rendered_mean_face)

    # get [zu zv z]
    lm_synthz_0 = lm_synthz_0.to(device).to(depth_0.dtype)

    # get the initial camera parameters
    K0 = pose.detach()[:, 16:].reshape(3, 3)
    R0 = pose.detach()[:, :16].reshape(4, 4) #cam2world; world2cam is R0^-1 
    lm_synthz_0[:,:2] = lm_synthz_0[:,:2] * depth_0[:, None]
    lm_synthz_0[:, 2] = depth_0

    lm_world = R0 @ torch.concat([torch.linalg.inv(K0) @ lm_synthz_0.T, torch.ones(1, 468).to(device)], dim=0)

    d_new = norm_init
    position_inv = -torch.matmul(torch.inverse(extrinsic[:3, :3]), extrinsic[:3, 3])
    position_inv_opt = position_inv.clone()
    pxz = d_new - (d_0 - position_inv[2])
    position_inv_opt[2] = position_inv[2] * pxz / position_inv[2]
    tz_0 = position_inv_opt[2].clone()
    position_update = torch.matmul(extrinsic[:3, :3], -position_inv_opt)
    focal_length_init = focal_length_init / d_0 * d_new
    position_init = position_update.clone()
    print(f"init distance with: {norm_init}, "
          f"position t_z with: {position_init[2]}, "
          f"focal {focal_length_init}")


    position_scale = 1
    lambda_x = 1.
    position_scale = torch.nn.Parameter(torch.tensor([position_scale],
                                    dtype=torch.float32, device=device, requires_grad=True))
    lambda_x = torch.nn.Parameter(torch.tensor([lambda_x],
                                    dtype=torch.float32, device=device, requires_grad=True))
    ortho_noise = torch.nn.Parameter(torch.zeros_like(ortho).requires_grad_(True).to(device))


    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    paras = [{"params": [lambda_x],
              "lr": hyperparameters.first_inv_cam_lr * hyperparameters.lambda_lr_scale},
             {"params": [ortho_noise],
              "lr": hyperparameters.first_inv_cam_lr * hyperparameters.rotation_lr_scale},
             {"params": [position_scale],
              "lr": hyperparameters.first_inv_cam_lr}]
    if use_lmd:
        paras.append({"params": [lmd_uncertainty],
                      "lr": hyperparameters.first_inv_cam_lr * hyperparameters.lmk_lr_scale})
    cam_optimizer = torch.optim.Adam(paras)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    CAM_FINISH = False
    noise_ramp_length = 1

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = (step - hyperparameters.camera_opt_step) / (num_steps - hyperparameters.camera_opt_step)
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)


        rot_opt = ortho2rotation(ortho + ortho_noise)
        pos_inv = -torch.matmul(torch.inverse(rot_opt), position_init)
        pos_inv_scale = pos_inv.clone()
        # pos_inv_scale[2] = pos_inv[2] * torch.sqrt(1 / torch.abs(position_scale))
        pos_inv_scale[2] = pos_inv[2] * position_scale
        position = torch.matmul(rot_opt, -pos_inv_scale)
        alpha = ((pos_inv_scale[2] - tz_0) + d_new) / d_new
        focal_length = focal_length_init * alpha * lambda_x
        c_opt = make_c(focal_length, rot_opt, position, device=device).unsqueeze(0)

        if hyperparameters.cam_opt_random_ws:
            synth = G.synthesis(ws, c_opt, noise_mode='const', force_fp32=True)
        else:
            synth = G.synthesis(w_opt, c_opt, noise_mode='const', force_fp32=True)

        synth_images = synth['image']
        synth_images = (synth_images + 1) * (255 / 2)

        #--------------------------------------------------        
        # Avoid call non-differentiable api during optimization
        # -------------------------------------------------
        # 1. get K
        K1 = c_opt[:, 16:].reshape(3, 3)
        # 2. get R
        R1 = torch.linalg.inv(c_opt[:, :16].reshape(4, 4))
        # 3. get new [u v z]
        lm_synthz_1 = (K1 @ ((R1 @ lm_world)[:3, :])).T
        lm_synthz_out = lm_synthz_1[:,:2] / lm_synthz_1[:,2][:, None]

        if use_lmd:
            imx = (synth_images.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)
            lmd_loss, w_sigmma = lm_loss(lm_synthz_out.unsqueeze(0), lm_target.unsqueeze(0), lmd_uncertainty)
            
            if hyperparameters.debug_dir is not None and hyperparameters.debug:
                os.makedirs(hyperparameters.debug_dir, exist_ok=True)
                # for visualization the landmark loss
                lm_synthz_np_1 = lm_synthz_out.detach().cpu().numpy()
                # sigmma = lmd_uncertainty.detach().cpu().numpy() ** 2
                sigmma = w_sigmma.detach().cpu().numpy()
                sigmma_norm = (sigmma - sigmma.min()) / (sigmma.max() - sigmma.min())
                for icc in range(468):
                    imx = cv2.circle(np.ascontiguousarray(imx), (int(lm_target_np[icc,0] * 512), int(lm_target_np[icc,1] * 512)), 
                                    radius=0, color=(255, 0, 0), thickness=5)
                    imx = cv2.circle(np.ascontiguousarray(imx), (int(lm_synthz_np_1[icc,0] * 512), int(lm_synthz_np_1[icc,1] * 512)), 
                                    radius=0, color=(0, 0, 255*sigmma_norm[icc]), thickness=5)
                cv2.imwrite(f'{hyperparameters.debug_dir}/vis_lmk_{step}.png', imx[:,:,::-1])

        else:
            lmd_loss = 0

        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        if border is not None:
            synth_images = synth_images[:, :, upper_b:down_b, left_b:right_b]
        elif mask_non_face is not None:
            synth_images = synth_images * mask_non_face

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization, from PTI
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # loss = dist + reg_loss * regularize_noise_weight + lmd_loss
        if step > hyperparameters.camera_opt_step:
            lmd_uncertainty.requires_grad_(False)
            CAM_FINISH = True
            loss = dist + reg_loss * regularize_noise_weight + lmd_weight * lmd_loss
        else:
            # dist_weight = step / hyperparameters.camera_opt_step
            dist_weight = 1
            # loss = dist * dist_weight + reg_loss * regularize_noise_weight + lmd_loss
            loss = dist * dist_weight + reg_loss * regularize_noise_weight + lmd_weight * lmd_loss

        with torch.no_grad():
            if use_wandb:
                wandb.log({f'Inversion_focal_{w_name}': focal_length.detach().cpu()}, step=global_config.training_step)
                wandb.log({f'Inversion_trans_{w_name}': position[2].detach().cpu()}, step=global_config.training_step)
                if use_lmd:
                    wandb.log({f'Inversion_lmd_{w_name}': lmd_loss.detach().cpu()}, step=global_config.training_step)

        if step % image_log_step == 0:
            with torch.no_grad():
                if use_wandb:
                    global_config.training_step += 1
                    wandb.log({f'first projection_{w_name}': loss.detach().cpu()}, step=global_config.training_step)
                    log_images_from_w([w_opt], [c_opt], G, w_name)

        # Optimization
        optimizer.zero_grad(set_to_none=True)
        cam_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step > hyperparameters.camera_opt_step:
            optimizer.step()
        cam_optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise, from PTI.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    with torch.no_grad():
        c_opt_ = c_opt.clone().detach()

    del G

    return w_opt, c_opt_
