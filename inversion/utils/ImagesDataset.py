import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.data_utils import make_dataset, get_border, load_mask


class ImagesDataset(Dataset):

    def __init__(self, source_root, name, source_transform=None, 
                 use_gt_c=False, mask_border=False, mask_non_face=False,
                 deep3d_path=None, unprocessed_dir=None):
        self.source_paths = sorted(make_dataset(source_root, name, use_gt_c=use_gt_c))
        self.source_transform = source_transform
        self.mask_border = mask_border
        self.mask_non_face = mask_non_face
        self.deep3d_path = deep3d_path
        self.unprocessed_dir = unprocessed_dir


    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, c_path, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        from_c = np.load(c_path)
        from_c = np.array(from_c, dtype=np.float32)
        if self.source_transform:
            from_im = self.source_transform(from_im)
        ori_img_path = os.path.join(self.unprocessed_dir, os.path.basename(from_path))

        assert not (self.mask_border and self.mask_non_face)
        if self.mask_border:
            border = get_border(from_path, ori_img_path=ori_img_path, deep3d_path=self.deep3d_path)
            return fname, from_im, from_c, border
        if self.mask_non_face:
            mask = load_mask(from_path)
            return fname, from_im, from_c, mask

        return fname, from_im, from_c, 0
