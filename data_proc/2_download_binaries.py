import os
import logging
import subprocess
import argparse
from pathlib import Path
from allstar import AllstarRepo
from datetime import datetime


class BinaryDownloader:
    '''
        This class (BinaryDownloader) extracts binaries from allstar repo.
        This class depends on allstar class and use the functions of it to communicate with allstar database (https://allstar.jhuapl.edu).
            
    '''
    def __init__(self, args, archs):
        logging.basicConfig(filename = datetime.now().strftime('%Y_%m_%d.log'), filemode='a', level=logging.INFO)
        self.logger = logging.getLogger()
        self.args = args
        self.archs = archs
        self.repos = [AllstarRepo(x) for x in self.archs]
        self.read_blacklist()
    
    def read_blacklist(self):
        f = open('blacklist.txt', 'r')
        self.blacklist_pkg = [pkg_name.strip().split(',')[0] for pkg_name in f.readlines()]
        f.close()

    def download_binaries(self, pkg_list):
        for pkg_name in pkg_list:
            if pkg_name in self.blacklist_pkg:
                continue

            for repo_i, repo in enumerate(self.repos):
                try:
                    bins = repo.package_binaries(pkg_name)
                except Exception as e:
                    print('**** Error %s with package (%s) %s ****' % (str(e), self.archs[repo_i], pkg_name))
                    continue

                for bin in bins:
                    output_root_dir_path = Path(self.args.output_dir).resolve()
                    bin_dir_path = output_root_dir_path / ("%s___%s-%s.bin" % (pkg_name, bin["name"], self.archs[repo_i])) 
                    bin_path = bin_dir_path / ("%s___%s-%s.bin" % (pkg_name, bin["name"], self.archs[repo_i]))  
                    url_info_path = bin_dir_path / 'url.txt'
                    
                    if bin_dir_path.exists():
                        print("skip processing %s in %s" % (bin['name'], bin_dir_path))
                        continue
                    bin_dir_path.mkdir(parents=True)

                    #TODO: it should be a bin_sizes.txt per process. :(
                    with open(bin_path, 'wb') as f:
                        self.logger.info("%s___%s-%s.bin size: %d" % (pkg_name, bin["name"], self.archs[repo_i], len(bin['content'])))
                        f.write(bin['content'])
                        
                    with open(url_info_path, 'w') as t:
                        t.write(bin['url'])


def extract_cpu_archs(pkg_list_file_name):
    return pkg_list_file_name.stem[19:].split("_")  # returning list of architectures i.e ['amd64', 'armel']


def extract_pkg_names(pkg_list_file_path):
    pkg_names = [] 
    with open(str(pkg_list_file_path), "r") as f:
        for pkg_name in f.readlines():
            pkg_names.append(pkg_name.strip())
    return pkg_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkg_list_path',type=str, default="/home/louisccc/NAS/louisccc/mindsight/pkg_lists/multiarch_pkg_list_amd64_armel_i386.txt", help='')
    parser.add_argument('--output_dir',type=str, default="/home/louisccc/NAS/louisccc/mindsight/global_trunk_4_cpus", help='Path to output folder.')
    args = parser.parse_args()

    pkg_list_file_path = Path(args.pkg_list_path).resolve()
    archs = extract_cpu_archs(pkg_list_file_path)
    pkg_names = extract_pkg_names(pkg_list_file_path)
    binary_dlr = BinaryDownloader(args, archs)
    binary_dlr.download_binaries(pkg_names)