Cfg2vec methodology for Function Name Reconstruction and Patch Situation
=====================
TreeEmbedding is a repository maintained by the UCI team for a Hierarchical Graph Neural Network (GNN) based approach for cross-architectural function name prediction.

![](https://github.com/AICPS/mindsight_cfg2vec/blob/6ae0a26c90ad2c639b925ac5029cfa6c9de789d0/archi.png)

## A. Environmental Setup

### Step 1. clone the repo and Create your Conda Working Environment
It is recommended to use the Anaconda virtual environment with Python 3.6. The guide for installing Anaconda on Linux is [here](https://docs.anaconda.com/anaconda/install/linux/). 
```sh
$ git clone https://github.com/AICPS/mindsight_cfg2vec.git
$ conda create --name [your env name] python=3.6
$ conda activate [your env name]
```
### Step 2. resolve Package Requirements 
This step was made based on a server with cuda 10.1 installed. You can also adjust the installation of torch and pyg according the hardware you have in your local (e.g., cpu or higher cuda).
```sh
$ cd mindsight_cfg2vec
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
$ python -m pip install torch-geometric==1.7.1
$ pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
$ python -m pip install -r requirements_cfg2vec.txt
$ conda install pygraphviz
```

## B. Playing (Train, Test, Evaluate) our `cfg2vec`

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

If you've followed Step 1 ~ Step 3, please make sure that you have the testing data in [./data/match_predict_test/](./data/match_predict_test/). And then, run the following commands for either function name prediction or function matching.
```sh
$ cd scripts/
# for app1: function matching task.
$ python app_cfg2vec.py --mode func_match --p1 ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --p2 ../data/toy_test/m-tx___prepmx-amd64.bin --pml "./saved_models/toy_train" --topk 10 --o result_fm.log --device cuda

# for app2: function name prediction task. 
$ python app_cfg2vec.py --mode func_pred --p ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --pdb toy_train.pkl --pml "./saved_models/toy_train" --topk 10 --o result_fpd.log --device cuda
```
With this command, you can refer to the prediction scores in the `./scripts/` after it is done.

You may also skip step 1~3 and use this provided pre-trained model. You may download the pre-trained model found [here](https://drive.google.com/file/d/1MClvWI8zh1TbNxwHVObUmtPu-huBgiKB/view?usp=sharing). This model is trained with binaries compiled for 3 architectures (amd64, armel, and i386). Then extract the model, and its corresponding pickle file into [script/](./scripts/) folder.

## Ackowledgements
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) and Naval Information Warfare Center Pacific (NIWC Pacific) under Contract Number N66001-20-C-4024. The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
