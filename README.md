Cfg2vec methodology for Function Name Reconstruction and Patch Situation
=====================
TreeEmbedding is a repository maintained by the UCI team for a Hierarchical Graph Neural Network (GNN) based approach for cross-architectural function name prediction.

![](https://github.com/AICPS/mindsight_cfg2vec/blob/6ae0a26c90ad2c639b925ac5029cfa6c9de789d0/archi.png)

## Environmental Setup

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

## Playing (Train, Test, Evaluate) our `cfg2vec`

### Step 1. prepare the training dataset
If you have your own binary dataset and want to run it with our code, please refer how to generate new training dataset's instructions in [./data_proc/README.md](/data_proc/README.md). However, this could take a much longer time. 

In this guide, we also provide a simple training dataset ([./toy_dataset.zip](./toy_dataset.zip)), which is a small subset of our derived *allStar dataset* mentioned in the paper. We suggest that try this out and see how our `cfg2vec` works first! Extract it right inplace.
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

### Step 3. Test and Evaluate our `cfg2vec`
The [`exp_cfg2vec_allstar.py`](scripts/exp_cfg2vec_allstar.py) script can be used to evaluate the trained model. We have provided a sample testing dataset to test the performance of our model. We have also provided a pre-trained model that can be downloaded for this purpose. The step-by-step guide to evaluating the model is:
1. Check if the test dataset (`toy_test`) is existed. This is an simple processed test dataset. If you have other binary, and want to test with it then you need preprocessing it by following instruction [here](/data_proc/README.md).
2. We also provide an pre-trained model which can be download from [here](https://drive.google.com/file/d/1MClvWI8zh1TbNxwHVObUmtPu-huBgiKB/view?usp=sharing). This model is trained with binaries compiled for 3 architectures (amd64, armel, and i386).

Extract the model
```python
$ tar xvf GoG_model.tar.gz
$ mv ./saved_models/GoG_train ./scripts/saved_models/GoG_train/
$ mv GoG_train.pkl ./scripts
$ rmdir saved_models/
```

3. Then you may use the following commands to evaluate the model. 
```python
$ cd scripts/
$ python exp_cfg2vec_allstar.py --dataset_path ../data/toy_train --pickle_path toy_train.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 --pml "./saved_models/toy_train"  --architectures 'armel, amd64, i386, mipsel'  --eval_only True --eval_dataset_path ../data/toy_test --eval_pickle_path toy_test.pkl
```
The prediction scores can be found in the [scripts/result](scripts/result) folder. 

## Testing for function name prediction
We made the script that utilizes the pre-trained cfg2vec model to run on Mindsight applications (function name prediction/function matching). The script can be run using either GPU or CPU. 
1. To run the script on a GPU:
    1. Follow the instructions on [Installation Guide](#Running_cfg2vec) and install the necessary packages if you haven't done so already.
    2. Train your own model, or if you want to use our pre-trained model, you may download the pre-trained model found [here](https://drive.google.com/file/d/1MClvWI8zh1TbNxwHVObUmtPu-huBgiKB/view?usp=sharing). This model is trained with binaries compiled for 3 architectures (amd64, armel, and i386). Then extract the model, and its corresponding pickle file into [script/](./scripts/) folder.
    3. We have sample testing data in [here](./data/match_predict_test/).
    4. Run the following commands for function name prediction or function matching.

    ```sh
    $ cd scripts/

    # for function matching task.
    $ python app_cfg2vec.py --mode func_match --p1 ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --p2 ../data/toy_test/m-tx___prepmx-amd64.bin --pml "./saved_models/toy_train" --topk 10 --o result_fm.log --device cuda

    # for function name prediction task. 
    $ python app_cfg2vec.py --mode func_pred --p ../data/toy_test/ipe5toxml___ipe5toxml-amd64.bin --pdb toy_train.pkl --pml "./saved_models/toy_train" --topk 10 --o result_fpd.log --device cuda
    ```
    The resulting log file can be found in the [scripts/](scripts/) folder.

## Ackowledgements
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) and Naval Information Warfare Center Pacific (NIWC Pacific) under Contract Number N66001-20-C-4024. The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
