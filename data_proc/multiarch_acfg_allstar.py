import subprocess
import argparse
from pathlib import Path
from allstar import AllstarRepo

class ASDatasetGenerator:
    '''
        This class (ASDatasetGenerator) extracts binaries from allstar repo and generates the datasets (ACFGs+Pcodes) .
        This class depends on allstar class and use the functions of it to communicate with allstar database (https://allstar.jhuapl.edu).
        Usage Example:
            python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support]
                --ghidra_proj [path to project folder, ~/treeEmbed1] --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] --range 0 10 
            python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support]
                --ghidra_proj [path to project folder, ~/treeEmbed1] --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] --range 10 20
            
    '''
    def __init__(self, args, archs):
        self.black_list = ["lib3dldflb.so.0.0.0", "libACE-6.2.8.so","ale", "a2jmidi_bridge", "a2jmidid", "j2amidi_bridge", "binutils-m68hc1x", "binutils-msp430"]
        self.black_list_pkg = ['algol68g', "ale", "binutils-m68hc1x", "binutils-msp430", "bear-factory", "cbmc", "ats-lang-anairiats-examples", "binutils-avr","connectome-workbench", "blt-dev", \
                                "connectome-workbench","ecasound-el", "djvulibre-bin","djvulibre-desktop", "iverilog", "chromium", "kicad-doc-it", "chuck", "lib32tinfo-dev",
                                "lib32nss-mdns", "lib32ncursesw5", "lib32ncursesw5-dev","lib32ncurses5", "lib32ncurses5-dev", "lemon", "gcc-avr", "ledger", "ledger-el", "ldc", "lcrack", "lcab",
                                "latex2rtf", "laserboy", "laserboy-indep", "lame", "kyotocabinet-utils", "kwin-style-dekorator", "kwave", "kuvert", "kst", "ksshaskpass", "ksplash-theme-tibanna", 
                                "ksplash-theme-bespin", "ksplash-theme-aperture", "kprinter4", "kpartsplugin", "korundum", "konwert", "konwert-filters", "konwert-dev", "komparator", "knutclient", 
                                "inetutils-tools","inetutils-traceroute","inetutils-ftpd","devrplay3","inetutils-ftp","freetype2-demos","extlinux","iptables","bibutils","asn1c","ion","inetutils-ping",
                                "inetutils-syslogd","eclib-tools","diod","ike","gcc-4.9-plugin-dev","inetutils-talkd","diffutils","blop","iptables-dev","clang-3.5-examples",
                                "freedv","composite","ike-qtgui","ecatools","gir1.2-listaller-0.5","avfs","cu","backfire-dkms","g++-4.8-multilib","fdutils","flex","inetutils-telnet",
                                "expeyes-doc-fr","genisoimage","inetutils-telnetd","kbd","hyantesite","drumstick-tools","cpp11-migrate-3.4","kdevelop-pg-qt","alsaplayer-oss","biosquid-dev",
                                "audiofile-tools","fluidsynth","eq10q","holdingnuts-server","ecasound","btag","info","fakechroot","elks-libc","inetutils-talk","kicad-doc-hu","d-itg",
                                "evemu-tools","biosquid","cpp11-migrate-3.5","bin86","findutils","bcrelay","kicad-doc-en","faketime","fcitx-ui-light","formed","clang-format-3.5",
                                "clang-3.4","clang-3.5","aqemu","holdingnuts","br2684ctl","fixincludes","iperf3","atm-tools","kicad-doc-de","cableswig","innoextract","alsa-utils",
                                "install-info","expeyes-firmware-dev","inetutils-inetd","gnuaisgui","gcc-4.8-locales","google-mock","alsaplayer-esd","eatmydata","alsaplayer-alsa",
                                "epub-utils","cutecom","alsaplayer-jack","cpp-4.8","g++-4.9-multilib","gmp-ecm","gtk2-engines-oxygen","cc1111","alsa-oss","fcitx-module-cloudpinyin",
                                "icedax","esekeyd","aiksaurus","kicad-doc-zh-cn","automoc","expeyes","atfs","flare-engine","gengetopt","fmit","clang-modernize-3.5","haveged","freemat",
                                "blender-ogrexml-1.8","cb2bib","cssc","ccrypt","freecad","freebsd-glue","bosixnet-webui","alsaplayer-nas","fcitx-frontend-qt5","expeyes-doc-en","kicad-doc-es",
                                "idn2","dicomnifti","bcc","alsaplayer-daemon","ats-lang-anairiats","hwinfo","kicad","kicad-doc-pl","chromedriver","gaiksaurus","alsoft-conf","hping3",
                                "blobby-server","gnuais","gcc-avr","cmake-qt-gui","bwctl-client","freemat-help","epm","bwctl-server","gcc-4.8-base","fplll-tools","fcitx-anthy","i2util-tools",
                                "ibus-chewing","fcitx-libs-qt5","fcitx-libs-qt5-dev","alsaplayer-xosd","fceux","hatari","gross","aqsis-examples","dolfin-bin","kicad-doc-pt","baycomepp",
                                "fcitx-chewing","blobby","clang-format-3.4","aqsis","kicad-doc-fr","atfs-dev","alsaplayer-gtk","clang-3.4-examples","clang-modernize-3.4","brewtarget",
                                "jimsh","fcitx-hangul","enum","aspcud","gperf","hdate","genders","ballerburg","expeyes-clib","dwarfdump","iec16022","calcurse","cde","gtk2-engines-qtcurve",
                                "fte-terminal","cwdaemon","gnucap","gcc-4.8","gcc-4.8-source","iaxmodem","clif","fcitx-frontend-fbterm","erlang-guestfs","kmetronome","fcitx-libpinyin","emu8051",
                                "fte-xwindow","cuba-partview","haskell98-report","fcitx-config-gtk2","gcc-4.9-source","frame-tools","fte","alsaplayer-text","kid3-qt","fte-console","ethtool",
                                "gtk3-engines-oxygen","dovecot-antispam","buici-clock","gamt","aseprite","autoclass","ibus-qt4","glhack","fte-docs","dvi2ps","kism3d","chipmunk-dev","eurephia",
                                "apvlv","freegish","gcc-4.8-multilib","fcitx-sayura","cyclades-serial-client","bppphyview","chemtool","fcitx-config-gtk","gcc-4.9-base","fakepop","kicad-doc-it",
                                "kid3-cli","chuck","cmake","freecad-dev","bosixnet-daemon","gcc-4.8-plugin-dev","inn","gcc-4.9","iwyu","coala","inotify-tools","cmake-curses-gui","hdparm","ebview",
                                "hepmc-examples","hepmc-reference-manual","cppad","kid3","g++-4.9","kid3-core","gcc-4.9-locales","amtterm","ats-lang-anairiats-examples","kicad-doc-ru",
                                "gcc-4.9-multilib","kmod","fcitx-unikey","cpp-4.9","fcitx-googlepinyin","dhcpcd-gtk","g++-4.8","knocker","hepmc-user-manual", "dirac", "idba", 
                                "coinor-libcoinutils-dev", "coinor-libcoinutils3", "extundelete", "exempi", "grap", "catcodec", "blackbox", "aptsh", "aldo", "icecc", "flrig", "boolstuff", "incron", "fstransform", "clustalw", "debian-xcontrol"
                                "aspell", "amule-emc", "ext3grep","geoip-bin", "clinfo", "cdargs", "ax25mail-utils", "boolstuff-dev","doscan", "freehdl", "fakeroot-ng"
        ]
        self.args = args
        self.archs = archs
        self.repos = [AllstarRepo(x) for x in self.archs]

    def run_ghidra_acfg_pcode_extraction(self, pkg_list):
        for pkg_name in pkg_list:
            if pkg_name in self.black_list_pkg:
                continue

            for repo_i, repo in enumerate(self.repos):
                try:
                    bins = repo.package_binaries(pkg_name)
                except Exception as e:
                    print('**** Error %s with package (%s) %s ****' % (str(e), self.archs[repo_i], pkg_name))
                    continue

                for bin in bins:
                    if bin['name'] in self.black_list:
                        continue
                    
                    print(bin['name'])
                    output_root_dir_path = Path(self.args.output_dir).resolve()
                    bin_dir_path = output_root_dir_path / ("%s___%s-%s.bin" % (pkg_name, bin["name"], self.archs[repo_i])) 
                    bin_path = bin_dir_path / ("%s___%s-%s.bin" % (pkg_name, bin["name"], self.archs[repo_i]))  
                    url_info_path = bin_dir_path / 'url.txt'
                    
                    if bin_dir_path.exists():
                        print("skip processing %s in %s" % (bin['name'], bin_dir_path))
                        continue
                    bin_dir_path.mkdir(parents=True)

                    with open(bin_path, 'wb') as f:
                        f.write(bin['content'])
                    
                    with open(url_info_path, 'w') as t:
                        t.write(bin['url'])

                    cmd_ghidra_headless = "%s/analyzeHeadless %s dummyProject%d -scriptPath %s -import %s -postScript DatasetGenerator.java %s -readOnly" % \
                                        (self.args.ghidra_path, self.args.ghidra_proj, repo_i, self.args.ghidra_scripts, bin_path, self.args.output_dir)
                    rc = subprocess.call(cmd_ghidra_headless, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pkg', type=int, default=20, help="The number of ")
    parser.add_argument('--pkg_list_path',type=str, default="./multiarch_pkg_list_amd64_armel_i386.txt", help='')
    parser.add_argument('--ghidra_path',type=str, default="~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support", help='Path to ghidra support folder.')
    parser.add_argument('--ghidra_proj',type=str, default="~/mindsight/", help='Path to ghidra project folder.')
    parser.add_argument('--ghidra_scripts',type=str, default="./ghidra_scripts_pcode", help='Path to ghidra scripts folder.')
    parser.add_argument('--output_dir',type=str, default="./temp", help='Path to output folder.')
    args = parser.parse_args()

    # Step 1: get a list of packages for multi-archi datasets. (options opened to any comb of archs) and then store it into offline files.
    pkg_list_path = Path(args.pkg_list_path).resolve()
    archs = pkg_list_path.stem[19:].split("_")
    pkg_list = []
    with open(str(pkg_list_path), "r") as f:
        for pkg_name in f.readlines():
            pkg_list.append(pkg_name.strip())
    
    # Step 2: TODO: subsample the list down to the desired size.
    import pdb; pdb.set_trace()

    # Step 3: use the resultant list to generate dataset accordingly.
    multiarch_repo = ASDatasetGenerator(args, archs)
    multiarch_repo.run_ghidra_acfg_pcode_extraction(pkg_list)