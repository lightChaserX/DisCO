import pickle
import torch
from utils.config_utils import get_checkpoints_dir

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def toogle_grad_3d(model, flag=True, conf=None):
    for name, param in model.named_parameters():
        if conf is not None:
            if hasattr(conf, 'only_generator') and conf.only_generator:
                if 'backbone.synthesis' in name:
                    param.requires_grad = flag
            else:
                param.requires_grad = flag
        else:
            param.requires_grad = flag


def load_tuned_G(filename, paths_config, device=torch.device('cuda:0')):
    new_G_path = get_checkpoints_dir(paths_config, filename)
    # import pdb; pdb.set_trace()
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G(paths_config, sampling_multiplier = 2, device=torch.device('cuda:0')):
    print(paths_config.stylegan2_ada_ffhq)
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device).eval()
        G = G.float()

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    G.rendering_kwargs['ray_start'] = 'auto'
    G.rendering_kwargs['ray_end'] = 'auto'
    return G


def load_old_D(paths_config, device=torch.device('cuda:0')):
    print(paths_config.stylegan2_ada_ffhq)
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_D = pickle.load(f)['D'].to(device).eval()
    old_D = old_D.float()
    return old_D
