import os
import torch
from tqdm import tqdm
from configs import global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import PIL.Image
import numpy as np
from utils.config_utils import get_embedding_dir, get_checkpoints_dir


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, paths_config, use_wandb, device="cuda:0"):
        super().__init__(data_loader, paths_config, use_wandb, device)

    def train(self):

        os.makedirs(f'{self.paths_config.experiments_output_dir}',exist_ok=True)
        use_ball_holder = True
       
        for fname, image, pose, border_or_mask in tqdm(self.data_loader):
            pose = pose.to(self.device)

            if hasattr(self.paths_config, 'mask_border') and self.paths_config.mask_border:
                mask_border = True
                if len(border_or_mask) == 1:
                    if border_or_mask[0] == 0:
                        mask_border = False
            else:
                mask_border = False
            if mask_border:
                print("Use ``masked border loss`` next")
                print("========================")
            else:
                print("Do NOT use ``masked border loss`` next")
                print("========================")
            
            if hasattr(self.paths_config, 'mask_non_face') and self.paths_config.mask_non_face:
                mask_non_face = True
                print("Use ``masked non-face loss`` next")
                print("========================")
            else:
                mask_non_face = False
                print("Do NOT use ``masked non-face loss`` next")
                print("========================")

            assert not (mask_non_face and mask_border), "mask_non_face and mask_border cannot be used together"
            if mask_border:
                border = border_or_mask
            else:
                border = None
            if mask_non_face:
                mask_binary = torch.tensor(border_or_mask).unsqueeze(0).to(self.device).float()
            else:
                mask_binary = None


            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= self.paths_config.max_images_to_invert:
                break

            embedding_dir = get_embedding_dir(self.paths_config, image_name)
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None
            if self.paths_config.use_last_w_pivots and os.path.isfile(f'{embedding_dir}/0.pt'):
                w_pivot, pose = self.load_inversions(embedding_dir, image_name)

            elif not self.paths_config.use_last_w_pivots or w_pivot is None:
                print("Calculate optimized parameters ...")
                w_pivot, pose = self.calc_inversions(image, pose, image_name, border=border, mask_binary=mask_binary)

                print("Save optimized parameters ...")
                print(pose)
                np.save(embedding_dir + '/cam_%s.npy'%(image_name), pose.squeeze().detach().cpu().numpy())
                torch.save(w_pivot, f'{embedding_dir}/0.pt')
           
            w_pivot = w_pivot.to(self.device)

            log_images_counter = 0
            real_images_batch = image.to(self.device)


            max_pti_steps = self.paths_config.max_pti_steps


            for i in tqdm(range(max_pti_steps+1)):
              
                generated_images, _ = self.forward(w_pivot, pose, eval=False)

                if mask_border:
                    upper_b = int(border['upper'])
                    down_b = int(border['down'])
                    left_b = int(border['left'])
                    right_b = int(border['right'])
                    loss, l2_loss_val, loss_lpips= self.calc_loss(generated_images[:, :, upper_b:down_b, left_b:right_b],
                                                                  real_images_batch[:, :, upper_b:down_b, left_b:right_b],
                                                                  image_name, self.G, use_ball_holder, pose, w_pivot)
                    
                elif mask_non_face:
                    ## todo: use masked perceptual loss
                    loss, l2_loss_val, loss_lpips= self.calc_loss(generated_images*mask_binary, real_images_batch*mask_binary, image_name,
                                                                  self.G, use_ball_holder, pose, w_pivot)
                else:
                    loss, l2_loss_val, loss_lpips= self.calc_loss(generated_images, real_images_batch, image_name,
                                                                  self.G, use_ball_holder, pose, w_pivot)


                self.optimizer.zero_grad()
                 
                if loss_lpips <= self.paths_config.LPIPS_value_threshold:
                  with torch.no_grad():
                    generated_images, _ = self.forward(w_pivot, pose, eval=True)
    
                    img = (generated_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.paths_config.experiments_output_dir}/{fname[0]}_{i:04d}'+'.png')

                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % self.paths_config.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], [pose], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            torch.save(self.G, get_checkpoints_dir(self.paths_config, image_name))
