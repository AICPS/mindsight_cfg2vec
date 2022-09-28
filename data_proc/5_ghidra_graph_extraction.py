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
        archi = bin_folder_name.split("-")[-1].split(".")[0]
        package = package_binary.split("___")[0]
        bin_name = package_binary.split("___")[1]
        if bin_name in all_bin_dict[package]:
            all_bin_dict[package][bin_name].append(archi)
        else:
            all_bin_dict[package][bin_name] = [archi]
    
    return all_bin_dict

def run_ghidra_acfg_pcode_extraction(pkg_list, args):
    process_num = int(args.ghidra_proj[-1])
    pkg_range = range((process_num - 1) * len(pkg_list)//4, (process_num) * len(pkg_list)//4)
    for i, (pkg_name, bins) in enumerate(sorted(pkg_list.items())):
        # import pdb; pdb.set_trace()
        if i in pkg_range:
            for bin_name, archs in bins.items():
                for arch in archs:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path = Path("%s/%s"%(args.root_dir, bin_dir_name))
                
                    cmd_ghidra_headless = "%s/analyzeHeadless %s dummyProject%d -scriptPath %s -import %s -postScript DatasetGenerator.java %s -readOnly" % \
                                        (args.ghidra_path, args.ghidra_proj, 0, args.ghidra_scripts, bin_dir_path, args.output_dir)
                    rc = subprocess.call(cmd_ghidra_headless, shell=True)

def check_proper_graph_extraction(all_bin_dict):
    # check if graph extract is properly done. if not then flag the pkg_name for removal.
    process_num = int(ghidra_proj[-1])
    pkg_range = range((process_num - 1) * len(pkg_list)//4, (process_num) * len(pkg_list)//4)
    flagged_pkg = set()
    for i, (pkg_name, bins) in enumerate(sorted(pkg_list.items())):
        if i in pkg_range:
            for bin_name, archs in bins.items():
                for arch in archs:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path = Path("%s/%s"%(args.root_dir, bin_dir_name))
                    bin_acfg_path = bin_dir_path / ("%s-acfg" % bin_dir_name)
                    if not bin_acfg_path.exists():                
                        flagged_pkg.add(pkg_name)
    return flagged_pkg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_pkg', type=int, default=20, help="The number of ")
    parser.add_argument('--root_dir',type=str, default="/home/louisccc/NAS/louisccc/mindsight/global_trunk_4_cpus_2", help='')
    parser.add_argument('--ghidra_path',type=str, default="~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support", help='Path to ghidra support folder.')
    parser.add_argument('--ghidra_proj',type=str, default="~/mindsight4", help='Path to ghidra project folder.')
    parser.add_argument('--ghidra_scripts',type=str, default="./ghidra_scripts_pcode", help='Path to ghidra scripts folder.')
    parser.add_argument('--output_dir',type=str, default="./temp", help='Path to output folder.')
    args = parser.parse_args()  

    all_bin_dict = get_all_bins_dict(args.root_dir)
    all_bin_list = sorted(all_bin_dict.items())
    # import pdb; pdb.set_trace()
    run_ghidra_acfg_pcode_extraction(all_bin_dict, args)
    
    # all_bin_dict.sort
    
    # import pdb; pdb.set_trace()



