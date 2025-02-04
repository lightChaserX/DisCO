import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
import lpips
from training.projectors import w_plus_orthcam_projector_media_alpha
from configs import global_config
from criteria import l2_loss
from utils.log_utils import log_image_from_w
from utils.models_utils import toogle_grad, toogle_grad_3d, load_old_G
import numpy as np

# Override the function
def fixed_normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.clamp(torch.sum(in_feat**2, dim=1, keepdim=True), min=eps))
    return in_feat / norm_factor

# Monkey patch LPIPS
lpips.normalize_tensor = fixed_normalize_tensor

class BaseCoach:
    def __init__(self, data_loader, paths_config, use_wandb, device="cuda:0"):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.paths_config = paths_config
        self.w_pivots = {}
        self.image_counter = 0
        self.device = device

        # Initialize loss
        self.lpips_loss = LPIPS(net=paths_config.lpips_type).to(self.device).eval()
        
        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = self.paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G(self.paths_config, device=self.device)
        toogle_grad_3d(self.G, True, conf=self.paths_config)

        self.original_G = load_old_G(self.paths_config, device=self.device)

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss, self.paths_config)
        self.optimizer = self.configure_optimizers()


    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{self.paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if self.paths_config.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not self.paths_config.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(self.device)
        return w_pivot

    def load_inversions(self, embedding_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name], None

        
        w_potential_path = f'{embedding_dir}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        
        pose = np.load(f'{embedding_dir}/cam_%s.npy'%(image_name))
        pose = torch.tensor(pose).to(self.device).unsqueeze(0)

        w = torch.load(w_potential_path).to(self.device)
        self.w_pivots[image_name] = w
        return w, pose

    def calc_inversions(self, image, pose, image_name, border=None, mask_binary=None):
        
        if self.paths_config.first_inv_type == 'w+_cam':
            id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
            pose = pose.to(self.device)
            w, pose = w_plus_orthcam_projector_media_alpha.project(self.G, id_image, pose, device=torch.device(self.device), w_avg_samples=600,
                                                             num_steps=self.paths_config.first_inv_steps, w_name=image_name,
                                                             use_wandb=self.use_wandb, hyperparameters=self.paths_config, border=border, mask_non_face=mask_binary)
        return w, pose

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=self.paths_config.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, pose, w_batch):
        loss = 0.0
        

        if self.paths_config.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * self.paths_config.pt_l2_lambda
        if self.paths_config.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * self.paths_config.pt_lpips_lambda

        if use_ball_holder and self.paths_config.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, pose, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w, pose, eval):
        if eval==True:
            self.G.eval()
        else:
            self.G.train()
        generated= self.G.synthesis(w, pose, noise_mode='const',force_fp32=True)
        generated_images = generated['image']
        generated_depths = generated['image_depth']
        return generated_images, generated_depths

