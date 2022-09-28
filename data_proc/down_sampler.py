import argparse
from collections import defaultdict
from pathlib import Path
from glob import glob
import shutil
import distutils
from distutils import dir_util
import random


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


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
    blacklist_pkg = [pkg_name.strip() for pkg_name in f.readlines()]
    f.close()
    return blacklist_pkg

def delete_flagged_pkgs(all_bin_dict, flagged_pkgs):
    for pkg_name, bins in all_bin_dict.items():
        if pkg_name in flagged_pkgs:
            for bin_name, archs in bins.items():
                bin_folder_paths = [Path("%s/%s___%s-%s.bin"%(args.root_dir, pkg_name, bin_name, arch)).resolve() for arch in archs]
                for bin_folder_path in bin_folder_paths: 
                    shutil.rmtree(str(bin_folder_path))


def check_too_many_bins(all_bin_dict, threshold=50):
    # check if there are too many bins for a single package.
    flagged_pkg = set()
    for pkg_name, bins in all_bin_dict.items():
        if len(bins) >= threshold:
            print(pkg_name, len(bins))
            flagged_pkg.add(pkg_name)
    return flagged_pkg


def only_bin_with_only_1_arch(all_bin_dict):
    # check if there is a binary with only 1 arch.
    flagged_pkg = set()
    for pkg_name, bins in all_bin_dict.items():
        for bin_name, archs in bins.items():
            if len(archs) == 1:
                print(pkg_name, bin_name)
                flagged_pkg.add(pkg_name)
    return flagged_pkg


def check_size0(all_bin_dict):
    # check if graph extract is properly done. if not then flag the pkg_name for removal.
    flagged_pkg = set()
    for pkg_name, bins in all_bin_dict.items():
        for bin_name, archs in bins.items():
            for arch in archs:
                bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                bin_dir_path = Path("%s/%s"%(args.root_dir, bin_dir_name))
                bin_path = bin_dir_path/bin_dir_name
                if bin_path.stat().st_size == 0:
                    flagged_pkg.add(pkg_name)
    return flagged_pkg

def check_proper_graph_extraction(all_bin_dict):
    # check if graph extract is properly done. if not then flag the pkg_name for removal.
    flagged_pkg = set()
    for pkg_name, bins in all_bin_dict.items():
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
    parser.add_argument('--root_dir',type=str, default="./temp", help='Path to output folder.')
    parser.add_argument('--out_dir',type=str, default="./temp2", help='Path to output folder.')
    parser.add_argument('--package_num',type=int, default=20, help='Num of packages to include.')
    parser.add_argument('--amd64', action='store_true', help='Toggle for including amd64.')
    parser.add_argument('--armel', action='store_true', help='Toggle for including armel.')
    parser.add_argument('--i386', action='store_true', help='Toggle for including i386.')
    parser.add_argument('--mipsel', action='store_true', help='Toggle for including mipsel.')
    args = parser.parse_args()  

    all_bin_dict = get_all_bins_dict(args.root_dir)

    
    selected_pkg = random.sample(list(all_bin_dict), args.package_num)

    for pkg_name in selected_pkg:
        bins = all_bin_dict[pkg_name]
        for bin_name, archs in bins.items():
            for arch in archs:
                if arch == 'amd64' and args.amd64:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path_src = Path("%s/%s"%(args.root_dir, bin_dir_name)).resolve()
                    bin_dir_path_dst = Path("%s/%s"%(args.out_dir, bin_dir_name))
                    bin_dir_path_dst.mkdir(exist_ok=True)
                    bin_dir_path_dst = bin_dir_path_dst.resolve()
                    dir_util.copy_tree(str(bin_dir_path_src), str(bin_dir_path_dst))
                if arch == 'armel' and args.armel:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path_src = Path("%s/%s"%(args.root_dir, bin_dir_name)).resolve()
                    bin_dir_path_dst = Path("%s/%s"%(args.out_dir, bin_dir_name))
                    bin_dir_path_dst.mkdir(exist_ok=True)
                    bin_dir_path_dst = bin_dir_path_dst.resolve()
                    dir_util.copy_tree(str(bin_dir_path_src), str(bin_dir_path_dst))
                if arch == 'i386' and args.i386:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path_src = Path("%s/%s"%(args.root_dir, bin_dir_name)).resolve()
                    bin_dir_path_dst = Path("%s/%s"%(args.out_dir, bin_dir_name))
                    bin_dir_path_dst.mkdir(exist_ok=True)
                    bin_dir_path_dst = bin_dir_path_dst.resolve()
                    dir_util.copy_tree(str(bin_dir_path_src), str(bin_dir_path_dst))
                if arch == 'mipsel' and args.mipsel:
                    bin_dir_name = "%s___%s-%s.bin"%(pkg_name, bin_name, arch)
                    bin_dir_path_src = Path("%s/%s"%(args.root_dir, bin_dir_name)).resolve()
                    bin_dir_path_dst = Path("%s/%s"%(args.out_dir, bin_dir_name))
                    bin_dir_path_dst.mkdir(exist_ok=True)
                    bin_dir_path_dst = bin_dir_path_dst.resolve()
                    dir_util.copy_tree(str(bin_dir_path_src), str(bin_dir_path_dst))

    
    # too_many_bin_pkgs = check_too_many_bins(all_bin_dict)
    # only_bin_with_only_1_arch_pkgs = only_bin_with_only_1_arch(all_bin_dict)   
    # non_proper_graph_extraction_pkgs = check_proper_graph_extraction(all_bin_dict)
    # size0_pkgs = check_size0(all_bin_dict)
    # import pdb; pdb.set_trace()
    # delete_flagged_pkgs(all_bin_dict, black_pkg_list)