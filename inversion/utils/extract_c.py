import json
from collections import OrderedDict
import glob
import shutil
import os
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# img_list = ['woman_perspective.jpg']

# with open('dataset.json', 'rb') as f:
#     json_data = json.load(f, object_pairs_hook=OrderedDict)
# labels = json_data["labels"]

def get_cam_labels(source_dir):
    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            print(labels)
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}
    return labels

cameras = get_cam_labels('Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/epoch_20_000000/')
# import pdb; pdb.set_trace()
print(cameras.keys())

image_list = list(glob.glob(os.path.join(args.indir, '**.png')))
image_list.extend(list(glob.glob(os.path.join(args.indir, '**.jpg'))))



print(image_list)
# print(img_list)
for i,img_path in enumerate(image_list):
    print(img_path)
    full_list = os.path.splitext(os.path.basename(img_path))
    image_name = full_list[0].split('.')[0]
    if os.path.exists(os.path.join(args.indir, 'mask_crop')):
        # considering multiple person
        multiple = True
        crop_dir = os.path.join(args.indir, 'crop/' + image_name + '_p**.png')
        crop_list = glob.glob(crop_dir)
        # import pdb; pdb.set_trace()
        if len(crop_list) > 0:
            image_name_list = [os.path.splitext(os.path.basename(ix))[0] for ix in crop_list]
        else:
            image_name_list = [image_name]
    else:
        image_name_list = [image_name]

    for image_name in image_name_list:
        print("processing:", image_name)
        if not os.path.exists(os.path.join(args.indir, 'crop/' + image_name + '.png')):
            print(f"skip image: {image_name} ==================================")
            continue
        # image_name_ori = full_list[0]
        image_name_ori = image_name
        image_key = image_name_ori+'.jpg'
        #image_key = img_path
        print(image_key)
        #     # id = int(image_name)*2
        label = cameras[image_key]
        print(image_key, cameras[image_key])
        np.save(os.path.join(args.indir, f'crop/{image_name}.npy'), np.array(label))
