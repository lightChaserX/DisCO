import os
import numpy as np
from PIL import Image
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def make_dataset(dir, name, use_gt_c=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if name is not None and name not in fname:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                if 'png' in path:
                   cpath = path.replace('png','npy')
                else:
                   cpath = path.replace('jpg','npy')
                if use_gt_c:
                    if 'png' in path:
                        cpath = path.replace('.png','_gt_c.npy')
                    else:
                        cpath = path.replace('.jpg','_gt_c.npy')
                if name is not None:
                    if fname == name:
                        print(fname)
                        images.append((fname, cpath,path))
                else:
                    images.append((fname, cpath,path))
    return images

#################################################
# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    im_size = np.asarray(img.size)
    # print("source size:", img.size)
    img = img.crop((left, up, right, below))
    # print("target:", left, up, right, below, img.size)

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                   t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
        np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    crop_xy = np.stack([left, up, right, below])

    return img, lm, mask, crop_xy.astype(np.int32), im_size.astype(np.int32)


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    print(lm3D, lm5p)
    print(f't and s: {t}, {s}')
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new, crop_xy, im_size = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)

    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    img = img.resize((w, h), resample=Image.BICUBIC)

    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new, crop_xy, im_size, img


def get_border(img_path, ori_img_path=None, deep3d_path=None):
    # TODO: replace it with pre-computed borders
    if deep3d_path is None:
        raise ValueError('Please provide the path to Deep3DFaceRecon_pytorch')
    sys.path.append(os.path.dirname(deep3d_path))
    sys.path.append(deep3d_path)
    from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
    lm3d_std = load_lm3d(f"{deep3d_path}/BFM/")
    
    if ori_img_path is None:
        ori_img_path = img_path.replace('/crop', '')
    if os.path.exists(ori_img_path):
        im = Image.open(ori_img_path).convert('RGB')
    else:
        im = Image.open(ori_img_path.replace('.png', '.jpg')).convert('RGB')
    lm_path = os.path.join(img_path.replace('/crop', '/detections').replace('.png', '.txt'))
    _,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    trans_params, im_high, _, _, crop_xy, im_size, _ = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)
    scale = trans_params[2]
    crop_xy_copy = crop_xy.copy()

    ############################################
    ## resize to 1024 to select the border
    ############################################
    # crop_xy_copy[3] = crop_xy_copy[3] - crop_xy_copy[1] if crop_xy_copy[1] < 0 else crop_xy_copy[3]
    # im_size[1] = im_size[1] - crop_xy_copy[1] if crop_xy_copy[1] < 0 else im_size[1]
    # crop_xy_copy[2] = crop_xy_copy[2] - crop_xy_copy[0] if crop_xy_copy[0] < 0 else crop_xy_copy[2]
    # im_size[0] = im_size[0] - crop_xy_copy[0] if crop_xy_copy[0] < 0 else im_size[0]
    # crop_xy_copy[1] = -crop_xy_copy[1] if crop_xy_copy[1] < 0 else 0
    # crop_xy_copy[0] = -crop_xy_copy[0] if crop_xy_copy[0] < 0 else 0
    #
    # up_0 = int(crop_xy_copy[1])
    # down_0 = int(min(min(im_size[1], crop_xy_copy[3]), target_size))
    # left_0 = int(crop_xy_copy[0])
    # right_0 = int(min(min(im_size[0], crop_xy_copy[2]), target_size))

    #=======================================================
    ## 1024 to select the border
    crop_xy_copy[3] = min(im_size[1], crop_xy_copy[3]) #h
    crop_xy_copy[2] = min(im_size[0], crop_xy_copy[2]) #w
    crop_xy_copy[3] = crop_xy_copy[3] - crop_xy_copy[1]
    crop_xy_copy[2] = crop_xy_copy[2] - crop_xy_copy[0]
    crop_xy_copy[1] = -crop_xy_copy[1] if crop_xy_copy[1] < 0 else 0
    crop_xy_copy[0] = -crop_xy_copy[0] if crop_xy_copy[0] < 0 else 0


    up_0    = int(crop_xy_copy[1])
    down_0  = int(min(crop_xy_copy[3], target_size))
    left_0  = int(crop_xy_copy[0])
    right_0 = int(min(crop_xy_copy[2], target_size))
    #=======================================================

    ############################################
    ## 1024 --> 700
    ############################################
    left = (im_high.size[0]/2 - center_crop_size/2)
    upper = (im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size

    ## border on 700
    left_1  = (max(left,  left_0 ) / center_crop_size * 512)
    upper_1 = (max(upper, up_0   ) / center_crop_size * 512)
    right_1 = (min(right, right_0) / center_crop_size * 512)
    lower_1 = (min(lower, down_0 ) / center_crop_size * 512)

    w_offset = ((center_crop_size/2 - im_high.size[0]/2) / center_crop_size * 512)
    h_offset = ((center_crop_size/2 - im_high.size[1]/2) / center_crop_size * 512)



    return {'left':  int(left_1+w_offset),
            'upper': int(upper_1+h_offset),
            'right': int(right_1+w_offset),
            'down':   int(lower_1+h_offset)}


def load_mask(img_path):
    """Load pre-computed mask image"""
    ori_img_path = img_path.replace('/crop', '/mask_crop')
    return (np.array(Image.open(ori_img_path).convert('L')) > 0) / 1.
