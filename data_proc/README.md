### Installation Guide: Generating Datasets

## To Get Started
It is recommended to use an Anaconda virtual environment and Python 3.6.
### 1. Create Anaconda Working Environment
It is recommended to use an Anaconda virtual environment and Python 3.6. Here we provide an installation script on a Linux system with CUDA 10.1. The guide for installing Anaconda in Linux is [here](https://docs.anaconda.com/anaconda/install/linux/) also.
```sh
$ conda create --name [your env name] python=3.6
$ conda activate [your env name]
$ python -m pip install pathlib==1.0.1
$ python -m pip install requests==2.24.0
```
### 2. Downloading Ghidra
From [Ghidra](https://github.com/NationalSecurityAgency/ghidra/releases), download release 10.0 titled ghidra_10.0_PUBLIC_20210621.zip. Once downloaded, unzip and move the Ghidra directory to the directory where you will be running this script. Install JDK11 from the oracle website [here](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) to your local machine. You will need this to open the GUI version of Ghidra.

### 3. Creating Projects in Ghidra
Within the Ghidra folder that you unzipped, there should be a executable titled ghidraRun. Run the executable and you will be met with the GUI version of Ghidra. Under the File tabin the top left, select New Project to create a new project. Select Non-Shared Project, give your project a name and click Finish. Locate the newly created Ghidra project folder in your system and copy the Ghidra project from your local working directory to where you will be running the script. Create as many copies of it as you would like. The number of copies will be the number of concurrent processes of the dataset generation script you will be able to run.

### 4. Running multiarch_acfg_allstar.py According to Your Task
Multiple processes of multiarch_acfg_allstar.py can be run simultaneously as long as there is a different Ghidra project folder for each instance of the script running at the same time. The range of packages may be split between the number of processes in any way. To run multiple processes, either create multiple terminals or use tmux. tmux also allows the scripts to continue to run even after the working environment is closed. An easy guide to tmux is [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).

Example 1: Allstar-200

20 packages 2 architectures
```sh
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed1] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 0 10
    --output_dir [path to output directory, ~/allstar-200-2]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed2] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 10 20
    --output_dir [path to output directory, ~/allstar-200-2]
```
Example 2: Allstar-300

20 packages 3 architectures
```sh
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed1] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 0 10 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-200-3]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed2] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 10 20 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-200-3]
```
Example 3: Allstar-2000

200 packages 2 architectures 
```sh
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed1] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 0 50
    --output_dir [path to output directory, ~/allstar-2000-2]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed2] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 50 100
    --output_dir [path to output directory, ~/allstar-2000-2]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed3] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 100 150
    --output_dir [path to output directory, ~/allstar-2000-2]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed4] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 150 200
    --output_dir [path to output directory, ~/allstar-2000-2]
```
Example 4: Allstar-3000

200 packages 3 architectures
```sh
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed1] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 0 50 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-2000-3]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed2] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode] 
    --range 50 100 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-2000-3]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed3] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode]
    --range 100 150 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-2000-3]
$ python multiarch_acfg_allstar.py --ghidra_path [path to ghidra, ~/ghidra_10.0_PUBLIC_20210621/ghidra_10.0_PUBLIC/support] 
    --ghidra_proj [path to project folder, ~/treeEmbed4] 
    --ghidra_scripts [path to scripts folder, ./ghidra_scripts_pcode]
    --range 150 200 
    --archs amd64 i386 armel
    --output_dir [path to output directory, ~/allstar-2000-3]
```