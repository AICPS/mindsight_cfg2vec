import os
import random
import sys
import pprint
from pathlib import Path

sys.path.append(os.path.dirname(sys.path[0]))
from core.models import cfg2vecCFG,cfg2vecGoG
import random
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from core.acfg_parser import ACFGDataset
from core.trainer import HSNTrainer
from torch_geometric.data import DataLoader


class Config():
    '''config for ACFG pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--dataset_path', type=str, default="../data/crossarch_top10_v2/acfg-data/", help="Path to dataset source folder.")
        
        self.p.add_argument('--mode', type=str, default="func_match", help="func_match/func_pred for function name matching/function name prediction")
        self.p.add_argument('--p', type=str, default="", help="Path to GoG features of binary.")
        self.p.add_argument('--p1', type=str, default="", help="Path to GoG features of binary 1.")
        self.p.add_argument('--p2', type=str, default="", help="Path to GoG features of binary 2.")
        self.p.add_argument('--pdb', type=str, default="./2cpu_20_2_cfg2vecCFG.pkl", help="Path to the pickle folder.")
        self.p.add_argument('--pml', type=str, default="./saved_models/cfg2vecCFG_2cpu_20_2", help="Path to the saved model.")
        
        self.p.add_argument('--batch_size', type=int, default=20, help='Number of graphs in a batch.')
        self.p.add_argument('--seed', type=int, default=random.randint(0,2**32), help='Random seed.')
        self.p.add_argument('--topk', type=int, default=10, help="k is used to decide how many candidate function names for each function name in the stripped binary.")
        

        self.p.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers to create for the model.')
        self.p.add_argument('--num_features', type=int, default=12, help="The initial dimension of each acfg node.")
        self.p.add_argument('--dropout', type=float, default=0.1, help='Dropout')
        self.p.add_argument('--layer_spec', type=str, default='16,16', help='String of dimensions for hidden layers.')
        self.p.add_argument('--device', type=str, default="cuda", help='The device to run on models, cuda is default.')
        self.p.add_argument('--metrics_path', type=str, default="./metrics/", help="Path to the metrics folder.")
        self.p.add_argument('--o', type=str, default="", help="Path to the result log.")

        self.p.add_argument('--learning_rate', default=0.001, type=float, help='The initial learning rate for GCN/GMN.')
        self.p.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        self.p.add_argument('--use_wandb', action='store_true', help='Use wandb')
        self.p.add_argument('--pcode2vec', type=str, default="none", help='[none|bb|func|bb_func]')
        self.p.add_argument('--wandb_project', type=str, default="cfg2vec", help='wandb project')

        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.dataset_path = Path(self.dataset_path).resolve()
        self.pml = Path(self.pml).resolve()
        self.metrics_path = Path(self.metrics_path).resolve()
        self.pdb = Path(self.pdb).resolve()
        self.p1 = Path(self.p1).resolve()
        self.p2 = Path(self.p2).resolve()
        self.p = Path(self.p).resolve()
        self.o = Path(self.o).resolve()


def get_dataset(path, num_features):
    dataset = ACFGDataset(cfg.pcode2vec)
    dataset.load_1_binary(str(path), num_features)
    dataset.pack4cfg2vec()
    return dataset 

def get_dataset_from_pickle(path):
    dataset = pd.read_pickle(str(path))
    dataset.pack4cfg2vec()
    return dataset 

def write_result(log_file, results):
    results_key = list(results.keys())
    for key in results_key:
        spec = key.split(".")
        log_file.write('{0:16}\n'.format("Package name  : {0}".format(spec[2])))
        log_file.write('{0:16}\n'.format("Binary file   : {0}".format(spec[3])))
        log_file.write('{0:16}\n'.format("Function name : {0}".format(spec[4])))
        log_file.write('{0:16}{1:19} {2:50}\n'.format("TopK matches  : ", "Similarity Score", "Function Name"))
        scores = list(results[key].keys())
        k = 0
        for score in scores:
            matched_names = results[key][score]
            for matched_name in matched_names:
                if k < cfg.topk:
                    log_file.write('{0:16}{1:>19} {2:50} \n'.format('', "{0}:".format(abs(score)), "{0}".format(matched_name)))
                    k+=1
                else:
                    break
            if k >= cfg.topk:
                break

if __name__ == "__main__":
    '''
        1. Usage:
            $ python app_cfg2vec.py --mode [func_match/func_pred] --p1 [path to 1st binary] --p2 [path to 2nd binary if mode == func_match] --k [number]

        2. Examples:
            $ python app_cfg2vec.py --mode func_match --p1 /media/NAS-temp/louisccc/mindsight/2CPU_dataset_20_2/autofs___automount-amd64.bin/ 
                                                      --p2 /media/NAS-temp/louisccc/mindsight/2CPU_dataset_20_2/autofs___automount-i386.bin/
                                                      --pml /media/NAS-temp/louisccc/mindsight/test_pretrained_model/cfg2vecCFG_2cpu_20_2/
                                                      --topk 10 --o result_fm.log
            $ python app_cfg2vec.py --mode func_pred --p /media/NAS-temp/louisccc/mindsight/2CPU_dataset_20_2/autofs___automount-amd64.bin/
                                                     --pdb /media/NAS-temp/louisccc/mindsight/test_pretrained_model/cfg2vecCFG_2cpu_20_2/2cpu_20_2_cfg2vecCFG.pkl
                                                     --pml /media/NAS-temp/louisccc/mindsight/test_pretrained_model/cfg2vecCFG_2cpu_20_2/
                                                     --topk 10 --o result_fpd.log
    '''

    cfg = Config(sys.argv[1:])

    if cfg.mode == "func_match":
        if not cfg.p1.exists() or not cfg.p2.exists():
            raise ValueError("The scripts expects two folders")

        dataloader1 = DataLoader(get_dataset(cfg.p1, cfg.num_features).end2end_dataset, batch_size=cfg.batch_size)
        dataloader2 = DataLoader(get_dataset(cfg.p2, cfg.num_features).end2end_dataset, batch_size=cfg.batch_size)
        
        model = cfg2vecGoG(cfg.num_layers, cfg.layer_spec, cfg.num_features, cfg.dropout, cfg.pcode2vec).to(cfg.device)
        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        results = trainer.do_func_matching(dataloader1, dataloader2, True)
        
        with open(str(cfg.o), "w") as log_file:
            write_result(log_file, results)
                
    elif cfg.mode == "func_pred":
        if not cfg.p.exists():
            raise ValueError("The scripts expects one folders")

        dataloader1 = DataLoader(get_dataset(cfg.p, cfg.num_features).end2end_dataset, batch_size=cfg.batch_size)
        dataloader2 = DataLoader(get_dataset_from_pickle(cfg.pdb).end2end_dataset, batch_size=cfg.batch_size)

        model = cfg2vecGoG(cfg.num_layers, cfg.layer_spec, cfg.num_features, cfg.dropout).to(cfg.device)
        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        results = trainer.do_func_matching(dataloader1, dataloader2, True)
        
        with open(str(cfg.o), "w") as log_file:
            write_result(log_file, results)
            

    else:
        raise ValueError("set mode to appropriate value: func_match/func_pred for function name matching/function name prediction")