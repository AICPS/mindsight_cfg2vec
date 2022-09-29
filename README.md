Cfg2vec methodology for Function Name Reconstruction and Patch Situation
=====================
TreeEmbedding is a repository maintained by the UCI team for a Hierarchical Graph Neural Network (GNN) based approach for cross-architectural function name prediction.

## To Get Started

<a name="Installation_Guide"></a>
### Installation Guide
It is recommended to use the Anaconda virtual environment with Python 3.6. The guide for installing Anaconda on Linux is [here](https://docs.anaconda.com/anaconda/install/linux/). 

#### 1. Clone the TreeEmbedding repository
```sh
$ git clone https://github.com/AICPS/mindsight_cfg2vec.git
```
#### 2. Create Anaconda Working Environment
```sh
$ conda create --name [your env name] python=3.6
$ conda activate [your env name]
```
#### 3. Install Necessary Packages
Once the environment is created, then install the following modules with the commands below:
```sh
$ cd mindsight_cfg2vec
$ export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
$ python -m pip install torch-geometric==1.7.1
$ pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
$ python -m pip install -r requirements_cfg2vec.txt
$ conda install pygraphviz
```

### training dataset
- Generated new training dataset with instructions in [./data_proc/README.md](/data_proc/README.md)
- we also provide simple training dataset
    - extracting the dataset 
    ```python
    $ tar xvf toy_dataset.tar.gz
    ```

## Training `cfg2vec`
The [`exp_cfg2vec_allstar.py`](scripts/exp_cfg2vec_allstar.py) python script is used to train the `cfg2vec` model. 
We used the *AllStar dataset*, which contains binary packages that are converted to a `CFG` representation.
With a combination of `CFG` and `GoG` Embedding layers, the model is trained to predict the `top-k` list of function names in a binary. The user may follow the steps below in order to carry out the aforementioned preprocessing, training, and evaluation.

For users who want to visualize the model training progress using [wandb](https://wandb.ai/site), we have enabled the support, and you just need to enable the option in the command line argument using `--use_wandb`. You may set the name of the wandb project using the `--wandb_project` option. 

General command:
```python
$ cd scripts/
$ python exp_cfg2vec_allstar.py --dataset_path [datasetpath] --pickle_path [.pkl file path] --seed 1 \
    --device cuda --epochs 100 --batch_size 4 --use_wandb --pml [path to model] \
    --wandb_project cfg2vec --architectures 'armel, amd64, i386, mipsel'
# to see more information about arguments
$ python exp_cfg2vec_allstar.py -h 
```

To train the model, you may refer to the following commands. The following command using the `toy_train`.
Example:
```python
# to run cfg2vec with the sample dataset
$ cd scripts/
$ python exp_cfg2vec_allstar.py --dataset_path ../toy_train --pickle_path toy_train.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 --pml "./saved_models/toy_train"  --architectures 'armel, amd64, i386, mipsel'
```

## To Evaluate the Trained Model
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
$ python exp_cfg2vec_allstar.py --dataset_path ../toy_train --pickle_path toy_train.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 --pml "./saved_models/toy_train"  --architectures 'armel, amd64, i386, mipsel'  --eval_only True --eval_dataset_path ../toy_test --eval_pickle_path toy_test.pkl
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

### Ackowledgements
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) and Naval Information Warfare Center Pacific (NIWC Pacific) under Contract Number N66001-20-C-4024. The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
