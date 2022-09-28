import os
from itertools import combinations
from allstar import AllstarRepo
from multiprocessing import Process


def all_binaries_exist(binaries):
    exist = True
    for b in binaries:
        if not b:
            exist = False
    return exist

def get_pkg_list_comb(comb):
    as_crawler = [AllstarRepo(x) for x in comb]
    base_packages = [x for x in as_crawler[0].package_list()]
            
    with open('multiarch_pkg_list_%s.txt'% ("_".join(comb)), 'w') as f:
        for pkg_name in base_packages:
            bins_all_arch = []
            for cralwer_idx, cralwer in enumerate(as_crawler):
                try:
                    binaries = cralwer.package_binaries_exist(pkg_name)
                    bins_all_arch.append(binaries)
                except Exception as e:
                    print('**** Error %s with package (%s) %s ****' % (str(e), comb[cralwer_idx], pkg_name))

            if not all_binaries_exist(bins_all_arch): # not all archs have this package in.
                continue              
            else:
                f.write("%s\n" % pkg_name)
                f.flush()
  

if __name__ == '__main__':
    
    all_cpu_archs = ["amd64", "armel", "i386", "mipsel"]
    
    all_processes = []
    
    for num_archs in range(2,len(all_cpu_archs)): # combination of 2~4

        for comb in combinations(all_cpu_archs, num_archs):

            print("Processing %s" % str(comb))
            p = Process(target=get_pkg_list_comb, args=(comb,))
            p.start()
            all_processes.append(p)

    for process in all_processes:
        process.join()