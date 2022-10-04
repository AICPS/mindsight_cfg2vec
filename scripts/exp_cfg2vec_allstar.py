'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.
'''

import os, random, sys
from typing_extensions import Self

sys.path.append(os.path.dirname(sys.path[0]))
from core.models import cfg2vecCFG, cfg2vecGoG
import pprint
from argparse import ArgumentParser
from pathlib import Path

import pickle as pkl
import pandas as pd
from core.acfg_parser import ACFGDataset
from core.trainer import HSNTrainer, MultiArchSampler
from torch_geometric.data import DataLoader
import numpy as np


class Config():
    '''config for ACFG pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--dataset_path', type=str, default="../data/crossarch_top10_v2/acfg-data/", help="Path to dataset source folder.")
        self.p.add_argument('--eval_dataset_path', type=str, default="../data/crossarch_top10_v2/acfg-data/", help="Path to evaluation dataset source folder.")
        self.p.add_argument('--pickle_path', type=str, default="acfg_graph.pkl", help="Path to the dataset pickle file.")
        self.p.add_argument('--eval_pickle_path', type=str, default="eval_dataset.pkl", help="Path to the dataset pickle file.")
        self.p.add_argument('--seed', type=int, default=random.randint(0,2**32), help='Random seed.')

        self.tg = self.p.add_argument_group('Training Config')
        self.tg.add_argument('--learning_rate', default=0.001, type=float, help='The initial learning rate for GCN/GMN.')
        self.tg.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        self.tg.add_argument('--batch_size', type=int, default=20, help='Number of graphs in a batch.')
        self.tg.add_argument('--device', type=str, default="cuda", help='The device to run on models, cuda is default.')
        self.tg.add_argument('--model', type=str, default="GOG", help="Model to be used intrinsically (GOG/CFG) GOG is default.")
        self.tg.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers to create for the model.')
        self.tg.add_argument('--num_features', type=int, default=12, help="The initial dimension of each acfg node.")
        self.tg.add_argument('--dropout', type=float, default=0.1, help='Dropout')
        self.tg.add_argument('--pml', type=str, default="./saved_models/")
        self.tg.add_argument('--test_step', type=int, default=10, help='Number of epochs before testing the model.')
        self.tg.add_argument('--layer_spec', type=str, default='16,16', help='String of dimensions for hidden layers.')
        self.tg.add_argument('--test_size', type=float, default=0.2, help='Test set size proportion if doing train-test split.')
        self.tg.add_argument('--architectures', type=str, default='i386,amd64,armel', help='String of architectures for parsing.')
        self.tg.add_argument('--tolerance', type=int, default=0, help="Tolerance count for early stopping.")
        self.tg.add_argument('--topk', type=int, default=10, help="topk.")
        self.tg.add_argument('--num_clusters', type=int, default=3, help='Number of clusters for clustering functions.')
        self.tg.add_argument('--pcode2vec', type=str, default="none", help='[none|bb|func|bb_func]')
        self.tg.add_argument('--use_wandb', action='store_true', help='Use wandb')
        self.tg.add_argument('--wandb_project', type=str, default="cfg2vec", help='wandb project')

        self.evaluate_group = self.p.add_argument_group('Evaluate Config')
        self.evaluate_group.add_argument('--eval_only', type=bool, default=False, help='Evaluate the model only (model must be loaded).')
        self.evaluate_group.add_argument('--metrics_path', type=str, default="./metrics/", help="Path to the metrics folder.")

        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.dataset_path = Path(self.dataset_path).resolve()
        self.eval_dataset_path = Path(self.eval_dataset_path).resolve()
        self.pickle_path = Path(self.pickle_path).resolve()
        self.eval_pickle_path = Path(self.eval_pickle_path).resolve()
        self.pml = Path(self.pml).resolve()
        self.metrics_path = Path(self.metrics_path).resolve()
        self.metrics_path.mkdir(exist_ok=True)
        

def read_dataset(cfg):
    if cfg.pickle_path.exists():
        dataset = pd.read_pickle(cfg.pickle_path)

    else:
        dataset = ACFGDataset(cfg.pcode2vec)
        dataset.load(cfg.dataset_path, cfg.num_features)
        # dataset.load(Path("/home/louisccc/NAS/louisccc/mindsight/darpa_challenges/dataset/"))
                
        with open(str(cfg.pickle_path), 'wb') as f:
            pkl.dump(dataset, f)
    
    dataset.pack4cfg2vec()
    return dataset

def read_eval_dataset(cfg):
    if cfg.eval_pickle_path.exists():
        dataset = pd.read_pickle(cfg.eval_pickle_path)
        
    else:
        dataset = ACFGDataset(cfg.pcode2vec)
        dataset.load(cfg.eval_dataset_path, cfg.num_features)
        # dataset.load(Path("/home/louisccc/NAS/louisccc/mindsight/darpa_challenges/dataset/"))
                
        with open(str(cfg.eval_pickle_path), 'wb') as f:
            pkl.dump(dataset, f)
    
    dataset.pack4cfg2vec()
    return dataset


if __name__ == "__main__":
    '''
        Usage:
            1.- calling end-to-end GAE approach on ACFG + ACG (allstar-7017)
                python exp_cfg2vec_allstar.py --dataset_path [path to allstar-7017 dataset, ~/NAS/louisccc/mindsight/allstar-7017/acfg-data/] --pickle_path allstar-7017.pkl --seed 1 
                                      --device cuda --num_layers 3 --layer_spec 32,32,32 --learning_rate 0.001 --model info --epochs 100 --batch_size 4

            2. calling end-to-end GAE approach on ACFG + ACG (Final Dataset)
                python exp_acfg_allstar.py --dataset_path /media/NAS-temp/louisccc/mindsight/final_dataset/ 
                --pickle_path final_dataset_cfg2vecGoG.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 
                --use_wandb True --pml "./saved_models/cfg2vecGoG_FinalDataset" 
                --architectures 'i386,amd64,armel,mipsel'

            3. Evaluation:
                python3 exp_acfg_allstar.py --dataset_path /media/aicps/home/louisccc/projects/debin/dataset/GoG_train --eval_only True 
                --pickle_path GoG_train.pkl --eval_dataset_path /media/aicps/home/vincent/TreeEmbedding/dataset/GoG_test_0 
                --seed 1 --device cuda --epochs 100 --batch_size 4 
                --pml "./saved_models/GoG_train" --architectures 'armel, amd64, i386'
    '''
    cfg = Config(sys.argv[1:])

    dataset = read_dataset(cfg)
    
    model = cfg2vecGoG(cfg.num_layers, cfg.layer_spec, cfg.num_features, cfg.dropout, cfg.pcode2vec).to(cfg.device)

    trainer = HSNTrainer(cfg, model, thunk_idx=dataset.func2idx['thunk']) # for siamese based network 
    
    if cfg.eval_only:
        print("Loading already trained model.")
        trainer.load_model()

        database, _ , _ = dataset.split_dataset_by_package(0, cfg.seed)
        database_data_loader = DataLoader(database, batch_size=cfg.batch_size)

        eval_set = read_eval_dataset(cfg)
        eval_set, _, _ = eval_set.split_dataset_by_package(0, cfg.seed)
        final_results = {}

        '''
        mipsel
        '''
        if "mipsel" in cfg.architectures:
            eval_results = {}
            mipsel_eval_set = [mipsel_bin for mipsel_bin in eval_set if mipsel_bin.archi == "mipsel"]
            eval_data_loader  = DataLoader(mipsel_eval_set,  batch_size=1, shuffle=True)
            eval_hits, eval_loss, eval_results = trainer.evaluate(eval_data_loader, database_data_loader, no_loss=True)
            
            # do evaluation.  
            for k in range(1, cfg.topk+1):
                eval_results['test_mipsel_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
                final_results['test_mipsel_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
            eval_results['test_mipsel_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            final_results['test_mipsel_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            # do function name prediction candidate list.
            if not os.path.exists('./result'):
                os.makedirs('./result')
            with open('./result/mipsel_tested_%s_model_%s.log'%(str(cfg.eval_dataset_path).split("/")[-1], str(cfg.pml).split("/")[-1]), "w") as log_file:
                pprint.pprint(eval_results, log_file)   
        ''' 
            arm 
        '''
        if "armel" in cfg.architectures:    
            eval_results = {}
            arm_eval_set = [arm_bin for arm_bin in eval_set if arm_bin.archi == "armel"]
            eval_data_loader  = DataLoader(arm_eval_set,  batch_size=1, shuffle=True)
            eval_hits, eval_loss, eval_results = trainer.evaluate(eval_data_loader, database_data_loader, no_loss=True)
            
            # do evaluation.  
            for k in range(1, cfg.topk+1):
                eval_results['test_arm_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
                final_results['test_arm_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
            eval_results['test_arm_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            final_results['test_arm_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            # do function name prediction candidate list.
            if not os.path.exists('./result'):
                os.makedirs('./result', 0o777)
            with open('./result/arm_tested_%s_model_%s.log'%(str(cfg.eval_dataset_path).split("/")[-1], str(cfg.pml).split("/")[-1]), "w") as log_file:
                pprint.pprint(eval_results, log_file)
        
        ''' 
            i386 
        '''
        if "i386" in cfg.architectures: 
            eval_results = {}
            i386_eval_set = [i386_bin for i386_bin in eval_set if i386_bin.archi == "i386"]
            eval_data_loader  = DataLoader(i386_eval_set,  batch_size=1, shuffle=True)
            eval_hits, eval_loss, eval_results = trainer.evaluate(eval_data_loader, database_data_loader, no_loss=True)
            # do evaluation.        
            for k in range(1, cfg.topk+1):
                eval_results['test_i386_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
                final_results['test_i386_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
            eval_results['test_i386_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            final_results['test_i386_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            # do function name prediction candidate list.
            if not os.path.exists('./result'):
                os.makedirs('./result', 0o777)
            with open('./result/i386_tested_%s_model_%s.log'%(str(cfg.eval_dataset_path).split("/")[-1], str(cfg.pml).split("/")[-1]), "w") as log_file:
                pprint.pprint(eval_results, log_file)
        ''' 
            amd64 
        '''
        if "amd64" in cfg.architectures: 
            eval_results = {}
            amd64_eval_set = [amd64_bin for amd64_bin in eval_set if amd64_bin.archi == "amd64"]
            eval_data_loader  = DataLoader(amd64_eval_set,  batch_size=1, shuffle=True)
            eval_hits, eval_loss, eval_results = trainer.evaluate(eval_data_loader, database_data_loader, no_loss=True)
            # do evaluation.        
            for k in range(1, cfg.topk+1):
                eval_results['test_amd64_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
                final_results['test_amd64_with_dataset-%s/p@%d'%(cfg.eval_dataset_path, k)] = eval_hits[k-1]
            eval_results['test_amd64_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            final_results['test_amd64_with_dataset-%s/loss'%cfg.eval_dataset_path] = eval_loss
            # do function name prediction candidate list.
            if not os.path.exists('./result'):
                os.makedirs('./result', 0o777)
            with open('./result/amd64_tested_%s_model_%s.log'%(str(cfg.eval_dataset_path).split("/")[-1], str(cfg.pml).split("/")[-1]), "w") as log_file:
                pprint.pprint(eval_results, log_file)
        
        if not os.path.exists('./result'):
            xos.makedirs('./result', 0o777)
        with open('./result/evaluation_%s_model_%s.log'%(str(cfg.eval_dataset_path).split("/")[-1], str(cfg.pml).split("/")[-1]), "w") as log_file:
            pprint.pprint(final_results, log_file)    

    else:    
        train_set, test_set, eval_set = dataset.split_dataset_by_package(cfg.test_size, cfg.seed)
                    
        train_data_loader = DataLoader(train_set, batch_sampler=MultiArchSampler(train_set))
        valid_data_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        test_data_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=True)
        print("Training new Model.")
        trainer.train_model(train_data_loader, valid_data_loader, test_data_loader)
        trainer.save_model()
