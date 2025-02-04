import sys 
sys.path.append("..")
sys.path.append("../..")
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from glob import glob

from tqdm import tqdm
import shutil
from configs import global_config
import wandb
import argparse

from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset
from utils.config_utils import load_config, get_embedding_dir, get_checkpoints_dir



def run_PTI(paths_config, 
            run_name='', 
            use_wandb=False, 
            use_multi_id_training=False, 
            file_name=None, 
            overwrite=None, 
            device='cuda:0'):

    global_config.run_name = run_name
    print(f"Run name is {global_config.run_name}")

    if os.path.exists(get_checkpoints_dir(paths_config, file_name)):
        if overwrite is not None and not overwrite:
            print(f"Skip {file_name} as it already exists")
            return None

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword,
                         dir=paths_config.wandb_dir,
                        reinit=True, name=global_config.run_name + '_' + file_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    
    embedding_dir_path = get_embedding_dir(paths_config, file_name)
    os.makedirs(embedding_dir_path, exist_ok=True)

    use_gt_c = True if hasattr(paths_config, 'use_gt_c') else False
    if use_gt_c:
        """For verfication. We will use ground truth camera for initialization"""
        print("We will use ground truth camera for initialization!")
    mask_border = True if hasattr(paths_config, 'mask_border') and paths_config.mask_border else False
    
    dataset = ImagesDataset(paths_config.input_data_path, file_name, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), use_gt_c, mask_border, 
        deep3d_path=paths_config.deep3d_path, unprocessed_dir=paths_config.unprocessed_folder)
    assert dataset.__len__() > 0


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, paths_config, use_wandb, device=device)
    else:
        coach = SingleIDCoach(dataloader, paths_config, use_wandb, device=device)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--example_config', type=str, help='Path to the example config file')
    argparser.add_argument('-f', '--file_name', type=str, default=None, nargs='?', help='Name of the file to run PTI on')
    argparser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files')
    argparser.add_argument('-w', '--wandb', action='store_true', help='Use wandb for logging')
    argparser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to run the training on')
    args = argparser.parse_args()

    
    example_config = args.example_config
    print(f"Using config file: {example_config}")
    paths_config = load_config(example_config)
    if not os.path.exists(paths_config.experiments_output_dir):
        os.makedirs(paths_config.experiments_output_dir)
    shutil.copy(example_config, f'{paths_config.experiments_output_dir}/config.py')

    file_list = glob(f'{paths_config.input_data_path}/**.png')
    file_list_name = [os.path.splitext(os.path.basename(f))[0] for f in file_list]

    if args.file_name is not None:
        file_list_name = [args.file_name]

    for file_name in tqdm(file_list_name):
        print(f"Processing {file_name}...")

        status = run_PTI(paths_config, 
                         run_name=paths_config.run_name, 
                         use_wandb=args.wandb, 
                         use_multi_id_training=False, 
                         file_name=file_name, 
                         overwrite=args.overwrite, 
                         device=args.device)

        