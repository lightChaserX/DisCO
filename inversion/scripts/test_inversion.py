import sys
sys.path.append("..")
sys.path.append("../..")
from utils.models_utils import load_tuned_G
from utils.eval_utils import MN, extract3d_landmark, get_eyes
from camera_utils import make_c
from configs import global_config
import argparse

import numpy as np
import torch
import mediapipe as mp

from glob import glob
import os
from utils.config_utils import load_config, get_embedding_dir
from utils.log_utils import to_img

from moviepy.editor import ImageSequenceClip


mp_face_mesh = mp.solutions.face_mesh
max_numoffaces = 1
bool_notracking = True



def test_PTI(face_mesh, 
             paths_config,  
             file_name=None, 
             dist_offset=0, 
             scale=None, 
             device="cuda:0",
             num_frames=40):

    
    embedding_dir = get_embedding_dir(paths_config, file_name)
    camera_path = embedding_dir + '/cam_%s.npy'% (file_name)
    face_path = f'{embedding_dir}/0.pt'
    print(camera_path)
    print(face_path)
    camera_np = np.load(camera_path)
    camera = torch.from_numpy(camera_np).to(device).unsqueeze(0)
    face = torch.load(face_path).to(device)

    extrinsic = camera.squeeze()[:16]
    intrinsic = camera.squeeze()[16:]
    rotation = extrinsic.reshape(4, 4)[:3, :3]
    position =  extrinsic.reshape(4, 4)[:3, 3]
    focal_length = (intrinsic[0] + intrinsic[4]) / 2

    G = load_tuned_G(file_name, paths_config, device=device)
    syn = G.synthesis(face.to(device), camera.to(device), noise_mode='const', force_fp32=True)

    pts3D, pts3D_rescale = extract3d_landmark(to_img(syn), face_mesh)
    pts3D_rescale = pts3D[:,:2].copy()
    pts3D_rescale[:, 0] = pts3D[:, 0]
    pts3D_rescale[:, 1] = pts3D[:, 1]
    lm_eye_right, lm_eye_left, lm_mouth_outer = get_eyes(pts3D_rescale)


    lmk_for_dist = np.concatenate((np.array([[(lm_eye_right[i, 1]), (lm_eye_right[i, 0])] for i in range(lm_eye_right.shape[0])]),
                                   np.array([[(lm_eye_left[i, 1]), (lm_eye_left[i, 0])] for i in range(lm_eye_left.shape[0])]),
                                   np.array([[(lm_mouth_outer[i, 1]), (lm_mouth_outer[i, 0])] for i in range(lm_mouth_outer.shape[0])])),
                                  axis=0)
    face_to_cam_dist = (syn["image_depth"].detach().cpu().numpy()[0, 0, (128 * lmk_for_dist[:, 0]).astype(np.int32),
                                                                  (128 * lmk_for_dist[:, 1]).astype(np.int32)]).mean()


    face_to_cam_dist += dist_offset
    print(face_to_cam_dist)
    print(syn["image_depth"].detach().cpu().numpy()[0, 0].max(), position[2].detach().cpu().numpy())

    dist_ratio_list = [scale]

    dist_ratio_list = (2 + torch.linspace(0, 14, num_frames)) / 2

    edit_cams = np.zeros((num_frames, 25))
    position_inv = -torch.matmul(torch.inverse(rotation), position)
    position_inv_opt = position_inv.clone()
    frames = []
    for i in range(num_frames):
        pxz = (face_to_cam_dist) * dist_ratio_list[i] - (face_to_cam_dist - position_inv[2])
        position_inv_opt[2] = position_inv[2] * pxz / position_inv[2]
        position_opt = torch.matmul(rotation, -position_inv_opt)
        focal_length_opt = focal_length * dist_ratio_list[i]
        print(position_opt, focal_length_opt)
        c_opt = make_c(focal_length_opt.unsqueeze(0), rotation, position_opt, device="cuda").unsqueeze(0)
        
        syn = G.synthesis(face, c_opt, noise_mode='const', force_fp32=True)
        save_folder = os.path.join(main_result_folder, file_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        img_path = save_folder + '/' + file_name + '_' + model_name +'_x' + str( dist_ratio_list[i].item()) + '_d' + str(dist_offset) + '.png'
        print(f"Check the result at {img_path}.")
        frames.append(np.array(to_img(syn, as_PIL=True)))
        if i == num_frames - 1:
            to_img(syn, as_PIL=True).save(img_path)
        edit_cams[i] = c_opt.detach().cpu().numpy()

    edit_camera_path = embedding_dir + '/edits_cam.npy'
    face_dist_path = embedding_dir + '/faces_dist.npy'
    lmk_det_path = embedding_dir + '/depths_pos.npy'
    np.save(edit_camera_path, edit_cams)
    np.save(lmk_det_path, lmk_for_dist)
    np.save(face_dist_path, np.array([face_to_cam_dist]))
    clip = ImageSequenceClip(frames, fps=12)
    clip.write_videofile(img_path.replace('.png', '.mp4'), fps=12)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example_config', type=str, help='Path to the example config file')
    parser.add_argument('-f', '--img_name', type=str, default=None, help='image name')
    parser.add_argument('-n', '--num_frames', type=int, default=40, help='number of frames')
    parser.add_argument('--d_offset', type=float, default=0, help='keep the plane with distance (eye+d_offset) fixed')
    parser.add_argument('--scale', type=int, default=8, help='scale the distance')
    args = parser.parse_args()

    example_config = args.example_config
    print(f"Using config file: {example_config}")
    paths_config = load_config(example_config)

    model_name = paths_config.pti_results_keyword

    if args.img_name is not None:
        file_list_name = [args.img_name]
    else:
        file_list = glob(os.path.join(paths_config.input_data_path, '**.png'))
        file_list_name = [os.path.splitext(os.path.basename(f))[0] for f in file_list]

    dist_offset = args.d_offset

    main_result_folder = paths_config.experiments_output_dir
    if not os.path.exists(main_result_folder):
        os.mkdir(main_result_folder)

    for file_name in file_list_name:

        with mp_face_mesh.FaceMesh(
                static_image_mode=bool_notracking,
                max_num_faces=max_numoffaces,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1) as face_mesh:

            test_PTI(face_mesh, paths_config,
                    file_name=file_name, 
                    dist_offset=dist_offset, 
                    scale=args.scale,
                    num_frames=args.num_frames)

