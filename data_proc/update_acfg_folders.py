'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.

    Distribution Statement "A" (Approved for Public Release,
    Distribution Unlimited) 
'''

# update_acfg_folders.py
import os
import logging
import subprocess
import argparse
from pathlib import Path
from allstar import AllstarRepo
from datetime import datetime
from collections import defaultdict
from glob import glob
import shutil
import distutils
from distutils import dir_util

def get_all_bins_dict(root_dir):
    
    all_bin_dict = defaultdict(dict)
    for bin_folder_name in list(glob("%s/**"%root_dir)):
        bin_folder_name = Path(bin_folder_name).name
        package_binary = "-".join(bin_folder_name.split("-")[:-1])
        # import pdb; pdb.set_trace()
        archi = bin_folder_name.split("-")[-1].split(".")[0]
        package = package_binary.split("___")[0]
        bin_name = package_binary.split("___")[1]
        if bin_name in all_bin_dict[package]:
            all_bin_dict[package][bin_name].append(archi)
        else:
            all_bin_dict[package][bin_name] = [archi]
    
    return all_bin_dict

def read_covered(file_name):
    f = open(file_name, 'r')
    covered_folders = [folder_name.strip() for folder_name in f.readlines()]
    print('covered list length:',len(covered_folders))
    f.close()
    return covered_folders

def copy_acfg_to_dest(root_dir, dest_dir):
    first_covered_folders = read_covered('/home/louisccc/NAS/louisccc/mindsight/9000_bin_two_batches/first_batch_covered.txt')
    second_covered_folders = read_covered('/home/louisccc/NAS/louisccc/mindsight/9000_bin_two_batches/second_batch_covered.txt')
    covered_lists = [first_covered_folders, second_covered_folders]
    dest_all_bins_dict = get_all_bins_dict(dest_dir)

    for i, flist in enumerate(covered_lists):
        for fname in flist:
            root_acfg_path = Path('%s/9000_bin_%d/%s/%s-acfg'%(root_dir, i+1, fname, fname))
            root_acfg_bb = Path('%s/9000_bin_%d/%s/%s-acfg/bb_pcode_attribute.csv'%(root_dir, i+1, fname, fname))
            root_acfg_pcode = Path('%s/9000_bin_%d/%s/%s-acfg/pcode_attribute.csv'%(root_dir, i+1, fname, fname))
            import pdb; pdb.set_trace()
            bin_folder_name = Path(root_acfg_path).name
            package_binary = "-".join(bin_folder_name.split("-")[:-1])
            archi = bin_folder_name.split("-")[-2].split(".")[0]
            package = package_binary.split("___")[0]
            bin_name = package_binary.split("___")[1].split("-")[0]
            if package in dest_all_bins_dict:
                if bin_name in dest_all_bins_dict[package]:
                    if archi in dest_all_bins_dict[package][bin_name]:
                        dest_acfg_path = Path('%s/%s/%s-acfg'%(dest_dir, fname, fname))
                        bb_path = shutil.copy(root_acfg_bb, dest_acfg_path)
                        pcode_path = shutil.copy(root_acfg_pcode, dest_acfg_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',type=str, default="/home/louisccc/NAS/louisccc/mindsight/9000_bin_two_batches", help='Path to acfg folders.')
    parser.add_argument('--dest_dir',type=str, default="/home/louisccc/NAS/louisccc/mindsight/global_temp", help='Path to destination folders.')
    args = parser.parse_args()

    

    dest_dir_all_bins_dict = get_all_bins_dict(args.dest_dir)
    copy_acfg_to_dest(args.root_dir, args.dest_dir)

    # Assume root_dir has directory structure root_dir/9000_bin and root_dir/9000_bin_2