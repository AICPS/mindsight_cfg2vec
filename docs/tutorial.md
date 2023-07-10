Cfg2vec: a Hierarchical Graph Neural Network (GNN) methodology for Cross-Architectural Binary Modeling Tasks
=====================
In today's rapidly evolving digital landscape, reverse engineering (RE) stands as an essential process in managing and maintaining legacy software in mission-critical systems. Over the decades, these systems become increasingly susceptible to new threats as the original source code, development environments, and maintenance support may no longer be available. As a result, Reverse Engineers (REs) must navigate the intricate process of disassembling and decompiling binaries into higher-level representations. However, this task is made significantly more complex due to factors such as stripped binaries, meaningless symbol names, and the understanding of the semantics of coding elements.

Moreover, the rising variety of hardware architectures introduces additional complexities when it comes to identifying code similarities across diverse CPU architectures. This process is both time-consuming and demanding, requiring a deep level of expertise and understanding. Given these challenges, it is clear that there is a significant need for an efficient tool that can support REs in this daunting task.

Here is where cfg2vec offers a helping hand. Cfg2vec is a humble attempt at addressing these intricate challenges that REs often face in their work. It uses a hierarchical Graph Neural Network (GNN) to reconstruct the names of each binary function, aiming to streamline the process of debugging and patching legacy binaries. By developing a Graph-of-Graph representation and combining Control Flow Graph (CFG) and Function Call Graph (FCG), cfg2vec seeks to model the relationship between binary functions' representations and their semantic names across different architectures. Though the task involves dealing with stripped binaries, understanding the semantics of coding elements, and bridging gaps across diverse CPU architectures, cfg2vec has been designed to help alleviate these burdens. It is our hope that through cfg2vec, REs will be better equipped to handle the challenges that come with maintaining legacy systems, contributing to the security and longevity of mission-critical software.

Building on the introduction to cfg2vec and its role in revolutionizing the landscape of reverse engineering, the following sections of this document aim to provide a comprehensive tutorial for users. <!--- Whether you're new to reverse engineering or a seasoned professional, we understand the importance of a clear, easy-to-follow guide to help you get the most out of the tool. -->
This tutorial is meticulously designed to provide step-by-step guidelines that walk you through the process of effectively utilizing cfg2vec. It covers everything from the basics of setting up the environment to training and performing function name prediction using CFG2VEC, allowing you to fully harness the power of this tool in your reverse engineering endeavors.

![](https://github.com/AICPS/mindsight_cfg2vec/blob/6ae0a26c90ad2c639b925ac5029cfa6c9de789d0/archi.png)

## A. Environmental Setup

### Step 1. Repository Cloning and Setting Up Your Conda Work Environment

We advise using the Anaconda virtual environment featuring Python 3.6 or above for optimal performance. Instructions for setting up your Anaconda environment on a Linux system can be found [here](https://docs.anaconda.com/anaconda/install/linux/). 
```sh
$ git clone https://github.com/AICPS/mindsight_cfg2vec.git
$ conda create --name [your env name] python=3.6
$ conda activate [your env name]
```
### Step 2. Installing Packages

Please note that the following steps are structured for a server equipped with CUDA 10.1. Depending on your local hardware specifications, such as CPU or higher CUDA versions, you may need to modify the installation of Torch and pyg accordingly.
```sh
$ cd mindsight_cfg2vec
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
$ python -m pip install torch-geometric==1.7.1
$ pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
$ python -m pip install -r requirements.txt
$ conda install pygraphviz
$ python -m pip install pathlib==1.0.1
$ python -m pip install requests==2.24.0
```
## B. Generating Datasets
The training and evaluation dataset for this tutorial was developed using the ALLSTAR (Assembled Labeled Library for Static Analysis Research) dataset, which is maintained by the Applied Physics Laboratory (APL). This expansive dataset includes over 30,000 pre-built Debian Jessie packages from i386, amd64, ARM, MIPS, PPC, and s390x CPU architectures, intended specifically for software reverse engineering research.

In the following section, we will provide a detailed, step-by-step guide on downloading the necessary binaries from the ALLSTAR dataset. We'll then show you how to preprocess these binaries to prepare the training and testing datasets.

### Step 1. Get list of packages that are available for download from the ALLSTAR database
In order to ensure the comprehensive functionality of our tool across different architectures, it is crucial that the binaries we utilize for training our model have corresponding counterparts in all the architectures that we are focusing on. For the sake of this tutorial, we will limit our focus to binaries compiled for the following four architectures:
1. amd64
2. armel
3. i386
4. mipsel

To maintain consistency and quality in our training dataset, it is vital that we possess binaries compiled for all the above-mentioned architectures. Fortunately, we have designed scripts to traverse the ALLSTAR datasets and perform sanity checks to ensure that this criterion is met.

We begin by running the Python script `1_generate_pkg_lists.py`, which will generate a list of packages available for download from the ALLSTAR database:
```python
$ python data_proc/1_generate_pkg_lists.py
```

This script will crawl through the ALLSTAR database for each architecture that we are focusing on and create a list of packages that exist across all these architectures. Let's delve a little deeper into how the script works. To optimize the execution of the script, the checks for each combination of architectures are performed in parallel processes, one for each combination of architectures.

Now, you will have a list of packages that are available across your target architectures, ready for download and subsequent processing.


### Step 2: DDownloading Packages from the ALLSTAR Database
This step is about downloading the packages from the ALLSTAR database that exist for all the architectures we are interested in (i.e., amd64, armel, i386, mipsel). We'll be using the package list generated in the previous step to download the corresponding binaries.

In the Python script `2_download_binaries.py`, you provide the path to the package list and the desired output directory for saving the downloaded binaries. The script extracts the list of packages from the file and begins the download process. The script also maintains a `'blacklist'` of packages to ignore during this process.

It's important to note that each downloaded binary will have a unique filename that follows this pattern: `pkg_name___binary_name-architecture.bin`. Along with each binary, a url.txt file is also generated, which stores the URL from where the binary was downloaded.

To initiate the downloading process, you would run the script with the appropriate command-line arguments like this:

```python
$ python data_proc/2_download_binaries.py --pkg_list_path /path/to/package/list/file --output_dir /path/to/output/directory
```

This script works by communicating with the ALLSTAR database to pull the required binaries for each package, across all specified architectures. Once all the binaries have been downloaded, you'll find them stored in the directory specified earlier.

The end result of this step is a collection of binaries compiled for various CPU architectures, which is ready to be used in the subsequent stages of the cfg2vec model pipeline.

### Step 3. Check and Filter Out Non-Compliant Packages

This step is about validating the downloaded binaries and filtering out the ones that do not comply with certain criteria. The purpose of this is to ensure that the training set will only include packages that are useful and meet the requirements. The steps performed in the script can be described as follows:

Check for Excessive Binaries: Packages with too many binaries could indicate potential issues or unnecessary complexity, so the script identifies these packages. The threshold of 'too many' can be defined.

Check for Single Architecture Binaries: Packages where some binaries are only available in a single architecture are flagged, as this could lead to imbalance in the training dataset.

Check for Large Binaries: Binaries exceeding a certain size threshold (in bytes) are flagged. This could help avoid memory issues during training or when running the model.

Check for Empty Binaries: The script also checks for binaries with a size of 0, which are clearly defective or incorrect and are therefore flagged.

Update the Blacklist: The identified packages from the above checks are added to the blacklist with the respective reasons for each package. This blacklist will be used to exclude these packages in the subsequent steps of the data processing pipeline.

To initiate this step, run the Python script with the appropriate directory as an argument:

```python
$ python check_filter_non_compliant_packages.py --root_dir /path/to/your/directory
```
This script logs any issues it encounters and adds non-compliant packages to the blacklist. These flagged packages will be excluded from subsequent steps of the process, ensuring only compliant packages are considered for model training.

### Step 4. Setting up Ghidra

### Step 5. Ghidra graph extraction

## C. Playing (Train, Test, Evaluate) our `cfg2vec`

### Step 1. prepare the training dataset
If you have your own binary dataset and want to run it with our code, please refer how to generate new training dataset's instructions in [./data_proc/README.md](/data_proc/README.md). However, this could take a much longer time. 

In this guide, we also provide a simple training dataset ([./toy_dataset.zip](./toy_dataset.zip)), which is a small subset of our derived *AllStar dataset* mentioned in the paper. We suggest that try this out and see how our `cfg2vec` works first! Extract it right inplace.
```python
$ unzip toy_dataset.zip
$ mkdir data
$ mv toy_train/ data/
$ mv toy_test/ data/
```

### Step 2. train our `cfg2vec` model
To train the `cfg2vec`, you can carry out the preprocessing, training, and evaluation with the following command:
```python
$ cd scripts/
$ python exp_cfg2vec_allstar.py --dataset_path [datasetpath] --pickle_path [.pkl file path] --device cuda --epochs 100 --batch_size 4 --use_wandb --pml [path to model] --architectures 'armel, amd64, i386, mipsel'
# an example using the `toy_train`
$ python exp_cfg2vec_allstar.py --dataset_path ../data/toy_train --pickle_path toy_train.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 --pml "./saved_models/toy_train" --architectures 'armel, amd64, i386, mipsel'
```
For more info regarding hyperparemeters and arguments, you can hit this command:
```
$ python exp_cfg2vec_allstar.py -h 
```

### Step 3. test/evaluate our `cfg2vec`
In previous step, our trainer script will perform mini-step testing during the training. However, if you specifically want this testing to be done standalone or toward one specific dataset, please consider follow these commands (we use `toy_test` as an example). 
```python
$ cd scripts/
$ python exp_cfg2vec_allstar.py --dataset_path ../data/toy_train --pickle_path toy_train.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 --pml "./saved_models/toy_train"  --architectures 'armel, amd64, i386, mipsel'  --eval_only True --eval_dataset_path ../data/toy_test --eval_pickle_path toy_test.pkl
```
With this command, you can refer to the prediction scores in the `scripts/result` after it is done. 

### Step 4: Running our `cfg2vec` for applicatons
To run our `cfg2vec` for applications, we've made the script utilizing the pre-trained cfg2vec model to run on **Function Name Prediction** and **Function Matching**.

If you've followed Step 1 ~ Step 3, please make sure that you have the testing data in [./data](./data/). And then, run the following commands for either function name prediction or function matching.
```sh
$ cd scripts/
# for app1: function matching task.
$ python app_cfg2vec.py --mode func_match --p1 ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --p2 ../data/toy_test/m-tx___prepmx-amd64.bin --pml "./saved_models/toy_train" --topk 10 --o result_fm.log --device cuda

# for app2: function name prediction task. 
$ python app_cfg2vec.py --mode func_pred --p ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --pdb toy_train.pkl --pml "./saved_models/toy_train" --topk 10 --o result_fpd.log --device cuda
```
With this command, you can refer to the prediction scores in the `./scripts/` after it is done.

You may also skip step 1~3 and use this provided pre-trained model. You may download the pre-trained model found [here](https://drive.google.com/file/d/1kAVwY_H4HPnRFThAu7sHfBj-5O93SsnB/view?usp=sharing). This model is trained with binaries compiled for 3 architectures (amd64, armel, and i386). 

Then extract the model, and its corresponding pickle file into [./scripts](./scripts/) folder.
```python
# If there is no `./scripts/saved_models` path please create one with following command 
# $ cd scripts/
# $ mkdir saved_models
# $ cd ..
$ unzip cfg2vec_pretrained_model.zip
$ mv GoG_train_8_12 ./scripts/saved_models/
$ mv GoG_train.pkl ./scripts/
$ cd scripts/
$ python exp_cfg2vec_allstar.py --pickle_path GoG_train.pkl --seed 10 --device cuda --epochs 100 --batch_size 4 --pml ./saved_models/GoG_train_8_12 --architectures "armel, amd64, i386" --eval_only True --eval_dataset_path ../data/toy_test --eval_pickle_path toy_test.pkl
```
If it returned "Killed" please check the memory usage. The program need 8GB free memory to read and store data from pickle file.

## Ackowledgements
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) and Naval Information Warfare Center Pacific (NIWC Pacific) under Contract Number N66001-20-C-4024. The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

Distribution Statement "A" (Approved for Public Release,Distribution Unlimited) 