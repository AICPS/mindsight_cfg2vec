import allstar
import os, sys, argparse
from io import BytesIO
from elftools.elf.elffile import ELFFile
from tqdm import tqdm
import requests.exceptions

# Imported only for error handling
import json
ERR_THRESHOLD = 10

class Config: 

    def __init__(self):
        # The core function of fetchAllstarRepo is fetch_allstar_repo(repo, dst_dir, binCmd, srcCmd)
        # Prerequisites to use fetch_allstar_repo includes:
        #       - declare directory location to write repo package folders and files
        #       - determine dispatch key (binCmd) based on desired selection for binary file download
        #       - set srcCmd bool flag to enable/disable source file download
        #       - initiate AllstarRepo object (defined in ./allstar.py)
        #
        # Dependant (less usual) libraries: 
        #       allstar.py - Library for interfacing with the ALLSTAR repo https://allstar.jhuapl.edu/
        #       pyelftools - ELF file analysis python library
        #       tqdm       - Progress bar library
        #       json       - Json library only used for error handling of allstar._package_index 
        #                     unsuccessful request.Session.get which is only exposed as a JSONDecodeError 
        #
        # Dependant local variables: 
        #       binariesToFetch - Dictionary mapping binCmd to binary fetch functions
        #       FetchStatistics - A python class for storing and tracking fetch_allstar_repo(...) counters
        #
        
        descript = '''DESCRIPTION:
        Generates a local ALLSTAR (sub)dataset by pulling the requested file type(s) from the ALLSTAR repo
        to the designated directory. Files types are defined and recognized as:\n
        \tunstripped/debug binary \tAn ELF file binary containing a .symtab 
                                \t\t and .debug_info/.zdebug_info section
        \tstripped binary         \tAn ELF file binary without a .symtab section
        \tsource code/file        \tA .c or .cpp file containing package source code\n
        Fetched files are saved into their designated file type subfolder generated within its associated
        package directory, creating the following folder hierarchy:\n
        \tbinary folder \t{DIRECTORY}/{package-name}_{ARCH}/bin/
        \tsource folder \t{DIRECTORY}/{package-name}_{ARCH}/src/ \n
        Where unstripped/debug binaries are stored with the '.bin' suffix and stripped binaries are stored
        with the '.sbin' file extension, uncategorized binaries are stored with the '.ubin' suffix. Empty 
        folders are removed as each package is done being searched for requested files.\n
        By default fetchAllstarRepo.py fetches only unstripped/debug binaries.'''

        examples = '''EXAMPLES:
        \tFetch unstripped binaries from repos {amd64 i386} to currenct directory:
        \t\tpython3 fetchAllStarRepo.py -d ./ -a amd64 i386\n
        \tFetch unstripped binaries and source code from repo {armel} to {/data/db/} directory:
        \t\tpython3 fetchAllStarRepo.py -c -d /data/db/ -a armel\n
        \tFetch stripped binaries from repos {amd64 armel i386 mipsel ppc64le s390x} to current directory:
        \t\tpython3 fetchAllStarRepo.py --stripped-only -d ./ -a amd64 armel i386 mipsel ppc64le s390x'''

        list_archs = '[amd64 armel i386 mipsel ppc64le s390x]'
        # Keep in mind the ALLSTAR website displays some architecture names differently than
        # how they are named in the database. Passing the website designated name will result
        # in the pull request terminating early for the specified architecture
        # 
        # Website HTML:
        #    <a href="armel/9mount/">armv7</a> 
        #    <a href="i386/9mount/">x86</a> 
        #    <a href="mipsel/9mount/">mipsel</a> 
        #    <a href="ppc64el/9mount/">ppc64le</a> 
        #    <a href="s390x/9mount/">s390x</a>
        #
        #       Website   Database
        #      --------------------
        #       armv7     armel
        #       x86       i386
        #       ppc64el   ppc64le

        self.parser = argparse.ArgumentParser(prog="python3 fetchAllstarRepo.py", 
                                        description=descript, 
                                        epilog=examples,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)

        group = self.parser.add_argument_group('REQUIRED ARGS')
        group.add_argument('-d','--directory',default='./',type=str,help='Output parent directory')
        group.add_argument('-a','--archs',action='store',dest='archs',type=str,nargs='*',default=["amd64"],
            help='List of repository architectures to fetch. All supported archs: '+ list_archs)

        group2 = self.parser.add_argument_group('OPTIONAL FETCH ARGS', 'Options related to which file types to fetch from ALLSTAR dataset')
        group2.add_argument('-w','--overwrite',action='store_true',help='Overwrite existing files when fetching repo')
        group2.add_argument('-s','--include-stripped',action='store_true',help='Include stripped binaries in repository fetch')
        group2.add_argument('-c','--include-source',action='store_true',help='Include source code in repository fetch')
        group2.add_argument('--stripped-only',action='store_true',help='Fetch only stripped binaries (no symtab)')
        group2.add_argument('--source-only',action='store_true',help='Fetch only source code')
        group2.add_argument('--unstripped-only',action='store_true',help='Fetch only unstripped binaries with debug info. (default behavior)')
        group2.add_argument('--all',action='store_true',help='Include all binaries and source code in repository fetch')

    def parse_args(self, input_args):
        args = self.parser.parse_args(input_args)

        # Check flags for conflicting logic
        ## Throw error if multiple "only" flags are declared
        ## Perform logical check if more than one arg is True
        if(args.unstripped_only 
            and (args.stripped_only or args.source_only) 
            or (args.stripped_only and args.source_only)):
            print("\nErr: Entered conflicting exclusive fetch flags\n")
            quit()

        ## Throw error if multiple "only" flags are declared
        if((args.include_stripped or args.include_source or args.all) 
            and (args.unstripped_only or args.stripped_only or args.source_only)):
            print("\nErr: Entered conflicting fetch flags\n")
            quit()

        # Verify directory arg is valid
        try:
            dst_dir = args.directory
            dst_dir = os.path.abspath(dst_dir)

            if not os.path.isdir(dst_dir):
                raise ValueError("Passed directory invalid")
            print("\nPulling ALLSTAR repo to %s\n"%dst_dir)
        except:
            print("\nErr: Entered invalid directory as an argument\n")
            quit()

        return args

class FetchStatistics:
    """ Fetch Statistics class was created to consolidate counters 
    when fetching the ALLSTAR repository. The class tracks number of files 
    downloaded, files saved, packages skipped, errors encountered, etc
    """
    def __init__(self, arch='', rsize = 0, binCmd=2):
        self.repo_size = rsize
        self.arch = arch
        self.last_pkg = ''
        self.binCmd = binCmd
        # Package stats
        self.skipped_pkg = 0
        self.empty_pkg = 0
        self.fetched_pkg = 0
        # Error stats
        self.errors = 0
        self.url_err = 0
        self.decode_err = 0
        self.http_err = 0
        self.timeout_err = 0
        # Binary stats
        self.dbg_bin = 0
        self.strp_bin = 0
        self.fetched_bin = 0
        self.found_bin = 0
        # Source stats   
        self.fetched_src = 0

    def countPackage(self):
        self.fetched_pkg += 1

    def countSkipPkg(self):
        self.skipped_pkg +=1

    def countEmptyPkg(self):
        self.empty_pkg +=1

    def countError(self):
        self.errors += 1

    def countUrlErr(self):
        self.url_err += 1

    def countDecodeErr(self):
        self.decode_err += 1

    def countHttpErr(self):
        self.http_err += 1

    def countTimeoutErr(self):
        self.timeout_err += 1
    
    def countSrc(self, numFetched):
        self.fetched_src += numFetched

    def countBin(self, fetched_bin, found_bin, dbg_bin, strp_bin):
        self.fetched_bin += fetched_bin
        self.found_bin += found_bin
        self.dbg_bin += dbg_bin
        self.strp_bin += strp_bin
    
    def printStats(self):
        bintype = {0: 'No', 1: 'Stripped', 2: 'Unstripped/Debug', 3: 'All'}

        print("\nFetch Statistics for '%s' repo"%self.arch)
        print("---------------------------------------------------")
        print("Repo size: %d\tPkg last fetched: %s"%(self.repo_size, self.last_pkg))
        print("\nPackage Stats:")
        print("  # Fetched: %d\t# Skipped: %d\t# Empty: %d"%(self.fetched_pkg, self.skipped_pkg, self.empty_pkg))
        print("  # HTTP Errors: %d\t# Timeout Errors: %d"%(self.http_err, self.timeout_err))
        print("  Failed JSON Requests: %d\t# Other Errors: %d"%(self.decode_err, self.errors))
        print("\nFile Stats:")
        print("  Source files downloaded: %d"%(self.fetched_src))
        print("\n  BinCmd[%d] = Requested %s Binaries"%(self.binCmd, bintype[self.binCmd]))
        print("  Binaries Encountered:")
        print("      Stripped = %d  Unstripped = %d  Total = %d"%(self.strp_bin, self.dbg_bin, self.found_bin))
        print("  Binaries downloaded: %d"%(self.fetched_bin))
        
def start_fetch_errorlog(dst_dir, binCmd, srcCmd, overwrite, archs):
    with open("fetchErrLog.txt", "w") as file:
        file.write("Error Log for fetchAllstarRepo.py")
        file.write("\nArguments generated:")
        file.write("\n\tDirectory: %s BinCmd: %d SrcCmd: %d Overwrite: %s"%(dst_dir, binCmd, srcCmd, overwrite))
        file.write("\n\tArchs: {}".format(archs))

def contains_debug(elf):
    """ Contains Debug replaces function has_dwarf_info in
    the pyelftools library which returns True if an .eh_frame 
    section is present. Since GCC includes .eh_frame by default, 
    this would be considered a false positive.

    'elf' - [obj] ELFFile object as defined by pyelftools

    Return True if the ELF binary contains an uncompressed
    (.debug_info) or compressed (.zdebug_info) debug
    information section
    """
    return bool(elf.get_section_by_name('.debug_info') or
            elf.get_section_by_name('.zdebug_info'))

def contains_symtab(elf):
    """ Contains .symtab

    'elf' - [obj] ELFFile object as defined by pyelftools

    Return True if the ELF binary contains a symbol table
    (.symtab), otherwise return False
    """
    return bool(elf.get_section_by_name('.symtab'))

def num_dwarf_cu(dwarfinfo):
    """ Number of Dwarf Compile Units

    'elf' - [obj] ELFFile object as defined by pyelftools

    Return True if the ELF binary contains more than 1
    DWARF compile unit, otherwise return False
    """
    num_cu = 0
    for compile_unit in dwarfinfo.iter_CUs():
        num_cu += 1
    return num_cu

def valid_debug_elf(elf):
    """ Valid Debug ELF

    'elf' - [obj] ELFFile object as defined by pyelftools

    Return True if the ELF binary contains
    a (z)debug_info section and has more than one DWARF/debug 
    compile unit, otherwise return False
    """
    if not contains_debug(elf):
        return False
        
    dwarfinfo=elf.get_dwarf_info()    
    if not (1 <= num_dwarf_cu(dwarfinfo)):
        return False

    return True

def fetch_pkg_stripped_binaries(pkg_bins, dst_dir):
    """ Fetch Package Stripped Binaries checks received package
    ELF binaries for those that dont contain a symbol table and 
    saves them in the designated folder with the '.sbin' file extension

    'pkg_bins' - [list[dict]] List containing received package binary files
    'dst_dir' - [str] Destination directory location to save downloaded files

    Return number of saved binary files, number of listed binary files, number
    of binary files with debug info, and number of stripped binary files
    """
    dbg_bin = 0
    strp_bin = 0
    fetched_bin = 0
    found_bin = 0

    for b in pkg_bins:
        elf = ELFFile(BytesIO(b['content']))
        found_bin += 1

        if not contains_symtab(elf):
            fetched_bin += 1
            strp_bin += 1
            binary_name = dst_dir + '/' + b['name'] + '.sbin'
            with open(binary_name, 'wb') as f:
                f.write(b['content'])
        else:
            dbg_bin += 1
            
    return fetched_bin, found_bin, dbg_bin, strp_bin

def fetch_pkg_debug_binaries(pkg_bins, dst_dir):
    """ Fetch Package Debug Binaries checks received package
    ELF binaries for those that contain valid debug information 
    and saves them in the designated folder with the '.bin' file extension

    'pkg_bins' - [list[dict]] List containing received package binary files
    'dst_dir' - [str] Destination directory location to save downloaded files

    Return number of saved binary files, number of listed binary files, number
    of binary files with debug info, and number of stripped binary files
    """
    dbg_bin = 0
    strp_bin = 0
    fetched_bin = 0
    found_bin = 0

    for b in pkg_bins:
        elf = ELFFile(BytesIO(b['content']))
        found_bin += 1
        if valid_debug_elf(elf):

            fetched_bin += 1
            dbg_bin += 1
            binary_name = dst_dir + '/' + b['name'] + '.bin'
            with open(binary_name, 'wb') as f:
                f.write(b['content'])
        else:
            strp_bin += 1

    return fetched_bin, found_bin, dbg_bin, strp_bin

def fetch_all_pkg_binaries(pkg_bins, dst_dir):
    """ Fetch All Package Binaries saves all received package
    ELF binaries with a file extension designating the "type". 
    ('.sbin': stripped binary (no symbol table), '.bin': binary w/
    debug info, '.ubin': undesignated binary (symbol table but no debug info))

    'pkg_bins' - [list[dict]] List containing received package binary files
    'dst_dir' - [str] Destination directory location to save downloaded files

    Return number of saved binary files, number of listed binary files, number
    of binary files with debug info, and number of stripped binary files
    """
    dbg_bin = 0
    strp_bin = 0
    fetched_bin = 0
    found_bin = 0

    for b in pkg_bins:
        elf = ELFFile(BytesIO(b['content']))
        found_bin += 1

        if not contains_symtab(elf):
            strp_bin += 1
            fext = '.sbin'
        elif valid_debug_elf(elf):
            dbg_bin += 1
            fext = '.bin'
        else:
            fext = '.ubin'
            #print("\nErr: Found uncharacterized binary %s{%s}\n"%(dst_dir,b['name']))

        fetched_bin += 1
        binary_name = dst_dir + '/' + b['name'] + fext
        with open(binary_name, 'wb') as f:
            f.write(b['content'])

    return fetched_bin, found_bin, dbg_bin, strp_bin

def fetch_pkg_source_code(pkg_src, dst_dir):
    """ Fetch Package Source Code saves the received source
    file data to corresponding '.cpp' files
    
    'pkg_src' - [list[dict]] List containing received package source files
    'dst_dir' - [str] Destination directory location to save downloaded files

    Return number of source files saved
    """
    fetched_src = 0

    for src in pkg_src:
        fetched_src += 1
        source_name = dst_dir + '/' + src['name'] + '.cpp'
        with open(source_name, 'wb') as f:
            for s in src['sources']:
                f.write(s['content'])

    return fetched_src

def check_overwrite(pkg_path, overwrite):
    """ Check Overwrite determines whether the downloading of package 
    binaries and/or source files should be skipped based on the 
    overwrite flag and if populated subdirectories exist within the 
    package folder

    'pkg_path'  - [str] Path location of the package folder
    'overwrite' - [bool] Overwrite flag indicating whether pre-existing 
                         files/directories should be overwritten

    Returns two booleans indicating whether fetch/download of package
    [0] binaries and/or [1] source files should be skipped 
    """
    skip_bins = False
    skip_src = False
    
    if overwrite:
        return (skip_bins, skip_src)

    bin_path = os.path.join(pkg_path, "/bin")
    if os.path.isdir(bin_path):
        if len(os.listdir(bin_path)):
            skip_bins = True

    src_path = os.path.join(pkg_path, "/src")
    if os.path.isdir(src_path):
        if len(os.listdir(src_path)):
            skip_src = True

    return (skip_bins, skip_src)

# Map binCmd to its related function
binariesToFetch = {
    #0: None,
    1: fetch_pkg_stripped_binaries,
    2: fetch_pkg_debug_binaries,
    3: fetch_all_pkg_binaries
}

def fetch_allstar_repo(repo, dst_dir, binCmd, srcCmd, overwrite=False):
    """ Fetch ALLSTAR Repo HTTP downloads the requested architecture 
    binary and/or source code files from allstar.jhuapl.edu.
    'repo'      - [obj] AllstarRepo object as defined by allstar.py
    'dst_dir'   - [str] Destination directory location to save downloaded files
    'binCmd'    - [int] Binary command key between 0 - 3 for binary 
                        download selection
    'srcCmd'    - [bool] Source command to enable (True) / disable (False) 
                         downloading source code files
    'overwrite' - [bool] Overwrite flag to enable (True) / disable (False)
                         pre-existing files/directories being overwritten 
                         during fetch process

    Return FetchStatistics object containing download and error stats
    """
    allstar_packages = repo.package_list()
    stats = FetchStatistics(repo.arch, len(allstar_packages), binCmd)

    repoBar = tqdm(total = len(allstar_packages), desc="Fetching %s repo..."%repo.arch, position=0)
    for pkg_name in allstar_packages:
    
        try:
            # Create folder ./dest_dir/pkg_arch/
            pkg_path = os.path.join(dst_dir, '{}_{}/'.format(pkg_name, repo.arch))
            
            # Create package directory if it does not exist
            if not os.path.isdir(pkg_path):
                os.mkdir(pkg_path)
                skip_bins = False
                skip_src = False
            else:
                # If overwrite is disabled skip fetch of binary/source files
                ## if the associated folders already exist
                (skip_bins, skip_src) = check_overwrite(pkg_path, overwrite)

            repoBar.set_description(desc="Fetching bins in %s from %s..."% (repo.arch, pkg_name))

            if(binCmd > 0) and not skip_bins:
                # Check number binaries in package by counting Request headers.
                ## This is also used to trigger possible JSONDecodeError before
                ## bin folder creation
                if(len(repo.package_binaries_exist(pkg_name))):
                    # Create folder ./dest_dir/pkg_arch/bin/
                    bin_path = os.path.join(pkg_path, "bin")
                    os.mkdir(bin_path)

                    # Request package binaries and dispatch to approprite
                    ## fetch_*_binaries function for file save selection
                    pkg_bins = repo.package_binaries(pkg_name)
                    binStats = binariesToFetch[binCmd](pkg_bins, bin_path)

                    # Save stats of encountered and saved binaries
                    stats.countBin(*binStats)

                    # delete /bin folder if empty
                    if not os.listdir(bin_path):
                        os.rmdir(bin_path)

            if(srcCmd) and not skip_src:
                # Request package source code files
                ## Call before src folder creation in case of JSONDecodeError
                pkg_src = repo.package_source_code(pkg_name)

                # Create folder ./dest_dir/pkg_arch/src/
                src_path = os.path.join(pkg_path, "src")
                os.mkdir(src_path)

                # Save (and count) package source files
                stats.countSrc(fetch_pkg_source_code(pkg_src, src_path))

                # delete /src folder if empty
                if not os.listdir(src_path):
                    os.rmdir(src_path)

        except requests.exceptions.InvalidURL:
            # Error thrown by allstar library when request in _package_index() 
            ## results in 404 response
            stats.countUrlErr()
            if(stats.url_err >= ERR_THRESHOLD):
                break
            pass

        except json.decoder.JSONDecodeError:
            # Count number of JSON decode errors. ALLSTAR contains some
            ## malformed json files.
            stats.countDecodeErr()
            pass

        except requests.exceptions.HTTPError:
            # Count and log HTTP Errors that occur in allstar.py
            ## TODO: should package folder be deleted since fetch
            ## might be incomplete?
            stats.countHttpErr()

            if(ERR_LOG):
                with open("fetchErrLog.txt", "a") as file:
                    file.write("\nHTTP Error: %s\n"%pkg_path)
                    file.write(str(sys.exc_info()))

            pass

        except requests.exceptions.Timeout:
            # Count and log timeout Errors that occur in allstar.py
            ## TODO: should package folder be deleted since fetch
            ## might be incomplete?
            stats.countTimeoutErr()

            if(ERR_LOG):
                with open("fetchErrLog.txt", "a") as file:
                    file.write("\nTimeout Error: %s\n"%pkg_path)
                    file.write(str(sys.exc_info()))

            pass

        except:
            # Increment number of Other Error occurrences. It is assumed all
            ## other errors are one offs that can be ignored. These errors
            ## are passed to prevent early termination of the fetch process
            stats.countError()

            if(ERR_LOG): 
                with open("fetchErrLog.txt", "a") as file:
                    file.write("\nError: %s\n"%pkg_path)
                    file.write(str(sys.exc_info()))

            pass

        finally:
            # delete /pkg_arch folder if empty
            if not os.listdir(pkg_path):
                os.rmdir(pkg_path)
                stats.countEmptyPkg()
            elif (binCmd > 0) and skip_bins:
                stats.countSkipPkg()
            else:
                stats.countPackage()
            
            stats.last_pkg = pkg_name
            repoBar.update(1)

    return stats


if __name__ == '__main__':
    ERR_LOG = True
    args = Config().parse_args(sys.argv[1:])

    dst_dir = os.path.abspath(args.directory)

    # Determine key value for binariesToFetch dispatch table
    ## #0: None,
    ## 1: fetch_pkg_stripped_binaries,
    ## 2: fetch_pkg_debug_binaries,
    ## 3: fetch_all_pkg_binaries
    if(args.all):
        ## Set key for 3: fetch_all_pkg_binaries
        binCmd = 3
    else:
        ## Determine which binary types should be pulled based on args flags
        ### Assume fetch debug binary, Unless stripped binaries only is declared
        bin_flags = not(args.stripped_only)

        ### Assume fetch stripped binary if either stripped flags declared, 
        #### Unless unstripped/debug binaries only is declared
        sbin_flags = (args.include_stripped | args.stripped_only) & ~(args.unstripped_only)
        
        ### Bitwise calculate binCmd dispatch key. 
        #### (Right) If source only is declared clear binary flags and set binCmd to 0
        #### (Left) If bin_flags is true shift to equal 2 then add 1 if sbin_flags is also true
        #### bin_flags * 2**1 + sbin_flags * 2**0 # multiply by 0 if source_only. Extend to 2 bits to
        ####         3: 11  2: 10  1: 01          # clear both (s)bin flags. True sets 00 False sets 11
        binCmd = (bin_flags << 1 | sbin_flags) & ~(args.source_only << 1 | args.source_only)

    # Assert fetch source files based if all or source flags declared
    srcCmd = args.all | args.include_source | args.source_only

    if(ERR_LOG): start_fetch_errorlog(dst_dir, binCmd, srcCmd, args.overwrite, args.archs)

    # Loop through each requested architecture for file download
    for arch in args.archs: 
        repo = allstar.AllstarRepo(arch)
        stats = fetch_allstar_repo(repo, dst_dir, binCmd, srcCmd, args.overwrite)
        stats.printStats()