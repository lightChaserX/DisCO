import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_name', type=str, default="tp0-MSBXWE", help='image name')
parser.add_argument('--config_file', type=str, default='example_configs/config.py', help='config file')
parser.add_argument('--d_offset', type=float, default=0, help='the distance from ')
parser.add_argument('--scale', type=int, default=8, help='scale the distance to new one')

args = parser.parse_args()

img_name = args.img_name
d_offset = args.d_offset
scale    = args.scale
#---------------------------------------------------------------------------------------------------------
print("run inversion=================================")
os.chdir('inversion/scripts')

cmd = f'python run_pti_vis -e ../../{args.config_file} -f {img_name} -o -w '
subprocess.run([cmd], shell=True, check=True)

#---------------------------------------------------------------------------------------------------------
print("test inversion=================================")
cmd = f'python test_inversion.py --example_config ../../{args.config_file} --img_name {img_name} --d_offset {d_offset} --scale {scale}'
subprocess.run([cmd], shell=True, check=True)


