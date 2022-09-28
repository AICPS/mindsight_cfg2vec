import os
import logging
import subprocess
import argparse
from pathlib import Path
from allstar import AllstarRepo
from datetime import datetime
from collections import defaultdict
from glob import glob

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
    f.close()
    return blacklist_pkg

def write_blacklist(pkg_list, reason):
    f = open('blacklist.txt', 'a')
    for pkg_name in pkg_list:
        f.write("%s,%s\n" % (pkg_name,reason))
    f.close()

def check_too_many_bins(all_bin_dict, threshold=50):
    # check if there are too many bins for a single package.
    flagged_pkgs = set()
    for pkg_name, bins in all_bin_dict.items():
        if len(bins) >= threshold:
            print(pkg_name, len(bins))
            flagged_pkgs.add(pkg_name)
    return flagged_pkgs

def only_bin_with_only_1_arch(all_bin_dict):
    # check if there is a binary with only 1 arch.
    flagged_pkgs = set()
    for pkg_name, bins in all_bin_dict.items():
        for bin_name, archs in bins.items():
            if len(archs) == 1:
                print(pkg_name, bin_name)
                flagged_pkgs.add(pkg_name)
    return flagged_pkgs

def check_size_large(all_bin_dict):
    # Threshold is 1,000,000 bytes
    threshold = 1000000
    flagged_pkgs = set()
    for pkg_name, bins in all_bin_dict.items():
        for bin_name, archs in bins.items():
            for arch in archs:
                bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                bin_dir_path = Path("%s/%s"%(args.root_dir, bin_dir_name))
                bin_path = bin_dir_path/bin_dir_name
                if bin_path.stat().st_size > threshold:
                    flagged_pkgs.add(pkg_name)
    return flagged_pkgs

def check_size_0(all_bin_dict):
    flagged_pkgs = set()
    for pkg_name, bins in all_bin_dict.items():
        for bin_name, archs in bins.items():
            for arch in archs:
                bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                bin_dir_path = Path("%s/%s"%(args.root_dir, bin_dir_name))
                bin_path = bin_dir_path/bin_dir_name
                if bin_path.stat().st_size == 0:
                    flagged_pkgs.add(pkg_name)
    return flagged_pkgs


if __name__ == '__main__':
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',type=str, default="/home/louisccc/NAS/louisccc/mindsight/global_trunk_4_cpus_2", help='')
    args = parser.parse_args()

    #   Get lists of packages that don't fit our requirements (too many binaries, only 1 arch, binary size too large, binary size 0)
    all_bin_dict = get_all_bins_dict(args.root_dir)

    too_many_bins = check_too_many_bins(all_bin_dict)
    print('Too many bins: ', too_many_bins)
    only_1_arch = only_bin_with_only_1_arch(all_bin_dict)
    print('Only 1 architecture: ', only_1_arch)
    too_large = check_size_large(all_bin_dict)
    print('Size too large: ', too_large)
    size_0 = check_size_0(all_bin_dict)
    print('Size 0: ', size_0)

    #   Add each list to the blacklist (with reason)
    #   TODO: remove duplicates?
    write_blacklist(too_many_bins, "too_many_bins")
    write_blacklist(only_1_arch, "only_1_arch")
    write_blacklist(too_large, "too_large")
    write_blacklist(size_0, "size_0")