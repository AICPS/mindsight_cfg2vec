import subprocess
import argparse


if __name__ == '__main__':
    '''
        python extract_graph.py --ghidra_path ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support \ 
           --ghidra_proj ~/mindsight --ghidra_scripts ./ghidra_scripts_pcode --output_dir ./temp \ 
           --bin_path ~/progs/program_c.clang.vuln
    '''
    parser = argparse.ArgumentParser()
          
    parser.add_argument('--ghidra_path',type=str,
        default="/home/kyle/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support",
        help='Path to ghidra support folder.')
    
    parser.add_argument('--ghidra_proj',type=str,
        default="/home/kyle/treeEmbed",
        help='Path to ghidra project folder.')
    
    parser.add_argument('--ghidra_scripts',type=str,
        default="./ghidra_scripts_pcode", 
        help='Path to ghidra scripts folder.')

    parser.add_argument('--output_dir',type=str,
        default="/home/kyle/allstar-188", 
        help='Path to output folder.')

    parser.add_argument('--bin_path',type=str,
        default="./xxx.bin",
        help='Path to bin file.')

    args = parser.parse_args()

    cmd_ghidra_headless = "%s/analyzeHeadless %s dummyProject%d -scriptPath %s -import %s -postScript DatasetGenerator.java %s -readOnly" % \
                            (args.ghidra_path, args.ghidra_proj, 0, args.ghidra_scripts, args.bin_path, args.output_dir)
    rc = subprocess.call(cmd_ghidra_headless, shell=True)    
