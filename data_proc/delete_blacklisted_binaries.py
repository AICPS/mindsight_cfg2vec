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

#   After finishing step 5, these functions will move to a separate file
def get_all_bins_dict(root_dir):
    
    all_bin_dict = defaultdict(dict)
    for bin_folder_name in list(glob("%s/**"%root_dir)):
        bin_folder_name = Path(bin_folder_name).name
        package_binary = "-".join(bin_folder_name.split("-")[:-1])
        archi = bin_folder_name.split("-")[-1].split(".")[0]
        package = package_binary.split("___")[0]
        bin_name = package_binary.split("___")[1]
        if bin_name in all_bin_dict[package]:
            all_bin_dict[package][bin_name].append(archi)
        else:
            all_bin_dict[package][bin_name] = [archi]
    
    return all_bin_dict

def read_blacklist():
    f = open('blacklist.txt', 'r')
    blacklist_pkg = [pkg_name.strip().split(',')[0] for pkg_name in f.readlines()]
    print('blacklist length:',len(blacklist_pkg))
    f.close()
    return blacklist_pkg

def delete_flagged_pkgs(all_bin_dict, flagged_pkgs):
    for pkg_name, bins in all_bin_dict.items():
        if pkg_name in flagged_pkgs:
            for bin_name, archs in bins.items():
                bin_folder_paths = [Path("%s/%s___%s-%s.bin"%(args.root_dir, pkg_name, bin_name, arch)).resolve() for arch in archs]
                for bin_folder_path in bin_folder_paths: 
                    shutil.rmtree(str(bin_folder_path))

if __name__ == '__main__':
    #   
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',type=str, default="./temp", help='Path to output folder.')
    args = parser.parse_args()  

    all_bin_dict = get_all_bins_dict(args.root_dir)
    print('before deleting, length:', len(all_bin_dict))
    black_pkg_list = read_blacklist()
    delete_flagged_pkgs(all_bin_dict, black_pkg_list)
    print('after deleting, length:', len(get_all_bins_dict(args.root_dir)))