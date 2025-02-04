from random import choice, randrange
from string import ascii_uppercase

from glob import glob
import shutil
import os
import json
import subprocess
import sys

from inversion.utils.config_utils import load_config

exts = ['png', 'jpg', 'jpeg']

def rename(start='x', offset=0):
    json_file = os.path.join(folder, 'original_name.json')
    if os.path.exists(json_file):
        return
    file_list = []
    for ext in exts:
        file_list.extend(glob(folder + '/*.' + ext))
    print(f"file numbers: {len(file_list)}")
    files_basename = [os.path.basename(f) for f in file_list]

    name_mapping = {}
    for i in range(len(files_basename)):
        i = i+offset
        ascii_str = ''.join(choice(ascii_uppercase) for i in range(6))
        rename = start + str(i) + '-' + ascii_str
        original_name = files_basename[i]
        if original_name.startswith(r'x+[0-9]'):
            continue
        ext_ = os.path.splitext(original_name)[1]
        name_mapping[rename+ext_] = original_name
        shutil.move(os.path.join(folder, original_name),
                    os.path.join(folder, rename + ext_))
        i += 1

    print(name_mapping)
    with open(json_file, 'w') as fp:
        json.dump(name_mapping, fp)

def unrename():
    json_file = os.path.join(folder, 'original_name.json')
    print(json_file)
    with open(json_file, 'r') as fp:
        print(fp)
        import pdb; pdb.set_trace()
        if len(fp.readlines()) != 0:
            fp.seek(0)
        name_mapping = json.load(fp)
    for name in name_mapping:
        shutil.move(os.path.join(folder, name),
                    os.path.join(folder, name_mapping[name]))


if __name__ == '__main__':
    paths_config = load_config(sys.argv[1])

    folder = paths_config.unprocessed_folder
    final_folder = os.path.dirname(paths_config.input_data_path).replace('../../', './') ## where the processed image
    if not folder.endswith('/'):
        folder += '/'

    rename_file_name = None
    if rename_file_name is not None:
        rename(rename_file_name)

    # TODO: move the file into the 
    eg3d_folder = paths_config.eg3d_dir ## eg3d folder
    code_folder = f'{eg3d_folder}/dataset_preprocessing/ffhq/'

    cmd = f"cp inversion/utils/extract_c.py {folder}"
    subprocess.run([cmd], shell=True, check=True)

    os.chdir(code_folder)
    cmd = f"python preprocess_in_the_wild.py --indir {folder}"
    subprocess.run([cmd], shell=True, check=True)

    cmd = f"python extract_c.py --indir {folder}"
    subprocess.run([cmd], shell=True, check=True)

    if not os.path.exists(f'{final_folder}/crop/'):
        os.makedirs(f'{final_folder}/crop/', exist_ok=True)
    cmd = f"mv {folder}/crop/* {final_folder}/crop/"
    subprocess.run([cmd], shell=True, check=True)

    if not os.path.exists(f'{final_folder}/detections/'):
        os.makedirs(f'{final_folder}/detections/', exist_ok=True)
    cmd = f"mv {folder}/detections/* {final_folder}/detections/"
    subprocess.run([cmd], shell=True, check=True)

