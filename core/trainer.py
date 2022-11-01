'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.

    Distribution Statement "A" (Approved for Public Release,
    Distribution Unlimited) 
'''

import itertools
import pprint
import sys
import warnings
import wandb
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm


warnings.filterwarnings('ignore') 

class MultiArchSampler(BatchSampler):
    def __init__(self, dataset):

        self.i386_list = []
        self.amd64_list = []
        self.mips_list = []
        self.arm_list = []

        self.binary_list = []

        for data_idx, data in enumerate(dataset): 
            if data.archi == "amd64":
                self.amd64_list.append(data_idx)
            elif data.archi == "i386":
                self.i386_list.append(data_idx)
            elif data.archi == "mipsel":
                self.mips_list.append(data_idx)
            elif data.archi == "armel":
                self.arm_list.append(data_idx)
            self.binary_list.append(data.package)
        self.tot_data = len(self.binary_list)

        self.i386_list = torch.tensor(self.i386_list)
        self.amd64_list = torch.tensor(self.amd64_list)
        self.mips_list = torch.tensor(self.mips_list)
        self.arm_list = torch.tensor(self.arm_list)

    def __iter__(self):
        perm_list = torch.randperm(self.i386_list.shape[0])
        perm_list_amd64 = torch.randperm(self.amd64_list.shape[0])
        perm_list_arm = torch.randperm(self.arm_list.shape[0])
        perm_list_mips = torch.randperm(self.mips_list.shape[0])
        shuffled_i386_list = self.i386_list[perm_list]
        shuffled_amd64_list = self.amd64_list[perm_list_amd64] 
        shuffled_arm_list = self.arm_list[perm_list_arm]
        shuffled_mips_list = self.mips_list[perm_list_mips] 

        for i386_data_idx in shuffled_i386_list: 
            data = [i386_data_idx.item()]
            arch_counterpart = np.where(np.array(self.binary_list) == self.binary_list[i386_data_idx])[0].tolist()
            arch_counterpart.remove(i386_data_idx)
            selected_idx = torch.tensor([1.0]*len(arch_counterpart)).multinomial(num_samples=1).item()
            data.append(arch_counterpart[selected_idx])
            
            same_bag_sel = torch.tensor([1.0]*self.tot_data)
            same_bag_sel[i386_data_idx] = 0.0 
            same_bag_idx = same_bag_sel.multinomial(num_samples=1).item()
            data.append(same_bag_idx)
            yield data
        
        for amd64_data_idx in shuffled_amd64_list: 
            data = [amd64_data_idx.item()]
            arch_counterpart = np.where(np.array(self.binary_list) == self.binary_list[amd64_data_idx])[0].tolist()
            arch_counterpart.remove(amd64_data_idx)
            selected_idx = torch.tensor([1.0]*len(arch_counterpart)).multinomial(num_samples=1).item()
            data.append(arch_counterpart[selected_idx])
            
            same_bag_sel = torch.tensor([1.0]*self.tot_data)
            same_bag_sel[amd64_data_idx] = 0.0 
            same_bag_idx = same_bag_sel.multinomial(num_samples=1).item()
            data.append(same_bag_idx)
            yield data
    
        for arm_data_idx in shuffled_arm_list: 
            data = [arm_data_idx.item()]
            arch_counterpart = np.where(np.array(self.binary_list) == self.binary_list[arm_data_idx])[0].tolist()
            arch_counterpart.remove(arm_data_idx)
            selected_idx = torch.tensor([1.0]*len(arch_counterpart)).multinomial(num_samples=1).item()
            data.append(arch_counterpart[selected_idx])
            
            same_bag_sel = torch.tensor([1.0]*self.tot_data)
            same_bag_sel[arm_data_idx] = 0.0 
            same_bag_idx = same_bag_sel.multinomial(num_samples=1).item()
            data.append(same_bag_idx)
            yield data

        for mips_data_idx in shuffled_mips_list: 
            data = [mips_data_idx.item()]
            arch_counterpart = np.where(np.array(self.binary_list) == self.binary_list[mips_data_idx])[0].tolist()
            arch_counterpart.remove(mips_data_idx)
            selected_idx = torch.tensor([1.0]*len(arch_counterpart)).multinomial(num_samples=1).item()
            data.append(arch_counterpart[selected_idx])
            
            same_bag_sel = torch.tensor([1.0]*self.tot_data)
            same_bag_sel[mips_data_idx] = 0.0 
            same_bag_idx = same_bag_sel.multinomial(num_samples=1).item()
            data.append(same_bag_idx)
            yield data

    def __len__(self):
        return len(self.i386_list)+len(self.amd64_list)

class HSNTrainer:
    '''hierarchical siamese network trainer.'''
    def __init__(self, cfg, model, thunk_idx=0):
        
        self.cfg = cfg
        self.minitest_metrics_filename = self.cfg.metrics_path / 'minitest_metrics.csv'
        self.cfg.metrics_path.mkdir(parents=True, exist_ok=True)
        self.thunk_idx = thunk_idx
        # random.seed(cfg.seed)
        # np.random.seed(cfg.seed)
        # torch.manual_seed(cfg.seed+1)
        # torch.random.manual_seed(cfg.seed+2)

        self.do_cluster = False
        self.model = model
        if cfg.use_wandb: # ask louis about thie key.
            self.wandb = wandb.init(settings=wandb.Settings(start_method='fork'), project=self.cfg.wandb_project, entity="aicpslab", config={}, name="picklePath_%s_numEpochs_%d_test_size_%d_seeds_%d"% (str(cfg.pickle_path)[47:-4], cfg.epochs, cfg.test_size, cfg.seed))
            self.wandb.watch(self.model, log="all")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.cfg.learning_rate, weight_decay=5e-4)

    def set_cluster_information(self, cluster_idx_test, cluster_idx_train, cluster_centers_test, cluster_centers_train):
        self.do_cluster = True
        self.cluster_idx_test = cluster_idx_test
        self.cluster_idx_train = cluster_idx_train
        self.cluster_centers_test = cluster_centers_test
        self.cluster_centers_train = cluster_centers_train

        loss_columns = ["train_loss", "test_loss"]
        columns = []
        columns += ["train_p_%d"%k for k in range(1, self.cfg.topk+1)]
        columns += ["test_p_%d"%k for k in range(1, self.cfg.topk+1)]
        cluster_columns = columns
        columns = loss_columns + columns
        for cluster in range(1, self.cfg.num_clusters+1):
            cluster_filename = self.cfg.metrics_path / f"{'cluster_minitest_metrics_.csv'}"
            pd.DataFrame(data=[], index=[], columns=cluster_columns).to_csv(cluster_filename)
        pd.DataFrame(data=[], index=[], columns=columns).to_csv(self.minitest_metrics_filename)

    def train_model(self, train_data_loader, valid_data_loader, test_data_loader):
        # early_stopper = EarlyStopper(self.cfg.tolerance)

        tqdm_bar = tqdm(range(self.cfg.epochs))

        for idx in tqdm_bar:
            final_loss = 0
            self.model.train()

            for batch in train_data_loader:
                #import pdb; pdb.set_trace()
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                batch = batch.to(self.cfg.device)
                acfg_graphs = list(itertools.chain.from_iterable(batch.features))
                acfgs = (next(iter(DataLoader(acfg_graphs, batch_size=len(acfg_graphs)))))
                acfgs = acfgs.to(self.cfg.device)
                if 'func' in self.cfg.pcode2vec:
                    acfg_embeds = self.model(acfgs.features, acfgs.edge_index, acfgs.batch, batch.edge_index, func_pcode_x=batch.func_pcode_features)
                else:
                    acfg_embeds = self.model(acfgs.features, acfgs.edge_index, acfgs.batch, batch.edge_index)
                acfg_labels = torch.LongTensor(list(itertools.chain.from_iterable(batch.function_labels))).to(self.cfg.device)
                               
                non_thunk_func_indexes = torch.where(acfg_labels!=self.thunk_idx)
                acfg_embeds = acfg_embeds[non_thunk_func_indexes]
                acfg_labels = acfg_labels[non_thunk_func_indexes]

                func_pair_gt_mtx = (acfg_labels.unsqueeze(0).repeat((acfg_embeds.size(0),1)) \
                                 == acfg_labels.unsqueeze(1).repeat((1,acfg_embeds.size(0))))
                
                func_pair_gt_mtx = func_pair_gt_mtx.int()
                func_pair_gt_mtx = 2*(func_pair_gt_mtx-0.5)
                func_pair_gt_mtx = torch.triu(func_pair_gt_mtx, diagonal=1)
                
                func_pair_gt_mtx = func_pair_gt_mtx.flatten()

                pos_sample_indexes = torch.where(func_pair_gt_mtx==1)
                pos_label_tensor = torch.full((len(pos_sample_indexes[0]),), 1, dtype=torch.float).to(self.cfg.device)

                neg_sample_indexes = torch.where(func_pair_gt_mtx==-1)
                neg_label_tensor = torch.full((len(neg_sample_indexes[0]),), -1, dtype=torch.float).to(self.cfg.device)
                try:
                    #import pdb; pdb.set_trace()
                    all_comb_a = acfg_embeds.unsqueeze(0).repeat((acfg_embeds.size(0),1,1)).view(-1,acfg_embeds.size(1))
                    all_comb_b = acfg_embeds.unsqueeze(1).repeat((1,acfg_embeds.size(0),1)).view(-1,acfg_embeds.size(1)) 

                    loss = 0
                    scalar = float(len(neg_sample_indexes[0])) / float(len(pos_sample_indexes[0]))
                    loss += scalar*F.cosine_embedding_loss(all_comb_a[pos_sample_indexes], all_comb_b[pos_sample_indexes], pos_label_tensor, 0.5, reduction='sum')
                    loss += F.cosine_embedding_loss(all_comb_a[neg_sample_indexes], all_comb_b[neg_sample_indexes], neg_label_tensor, 0.5, reduction='sum')
                
                    loss.backward()
                    final_loss += loss.detach().item()/(len(pos_sample_indexes)+len(neg_sample_indexes))
                    self.optimizer.step()
                except:
                    #import pdb; pdb.set_trace()
                    print(batch, batch.package, batch.archi)
                    print(len(batch.function_names[0]))
                    print(len(batch.function_names[1]))
                    print(len(batch.function_names[2]))
                    pass

            tqdm_bar.set_description('Epoch: {:04d}, loss: {:.4f}'.format(idx, final_loss/len(train_data_loader)))

            if idx % (self.cfg.test_step) == 0:

                if self.do_cluster:
                    # _, train_hits, train_hits_cluster, train_loss = self.cluster_inference(train_data_loader, self.cluster_idx_train)
                    _, test_hits, test_hits_cluster, test_loss = self.cluster_inference(test_data_loader, self.cluster_idx_test)

                else:
                    # func_embeds_train, names_train, train_hits, train_loss, train_results = self.inference(valid_data_loader)
                    func_embeds_test, names_test, test_hits, test_loss, test_results  = self.inference(test_data_loader)
                    eval_results = {}
                    '''for k in range(1,self.cfg.topk+1):
                        eval_results['train/p@%d'%k] = train_hits[k-1]
                    eval_results['train/loss'] = train_loss'''
                    for k in range(1,self.cfg.topk+1):
                        eval_results['test/p@%d'%k] = test_hits[k-1]
                    eval_results['test/loss'] = test_loss
                    self.wandb_or_print_results(eval_results)
                    
                    '''with open(self.cfg.metrics_path / ('trainset_.log'), "w") as log_file:
                        pprint.pprint(train_results, log_file, width=300)'''

                    with open(self.cfg.metrics_path / ('testset_.log'), "w") as log_file:
                        pprint.pprint(test_results, log_file, width=300)
                    
                    embed_filename = self.cfg.metrics_path / ('func_embeds_epoch_.csv')

                    # func_embeds = torch.cat([func_embeds_train, func_embeds_test], dim=0)
                    func_embeds = torch.cat([func_embeds_test], dim=0)

                    # func_names = names_train + names_test
                    func_names = names_test
                    columns = ["Feature_%d"%idx for idx in range(func_embeds.shape[1])]
                    pd.DataFrame(data=func_embeds.cpu().numpy(), index=func_names, columns=columns).to_csv(embed_filename)

                # print("Exporting %s" % self.minitest_metrics_filename)
                # results = [(train_loss/len(data_loader.dataset)), (test_loss/len(test_data_loader.dataset))]
                # results += train_hits # training p@1~p@5
                # results += test_hits # testing p@1~p@5
                # for i in range(0, self.cfg.num_clusters):
                #     self.export_cluster_csv(i+1, train_hits_cluster[i], test_hits_cluster[i])
                # pd.DataFrame(data=[results], index=[idx]).to_csv(self.minitest_metrics_filename, mode='a', header=False)

                    self.save_model()
                # if early_stopper.should_stop(test_loss/len(test_data_loader.dataset)):
                #     break
    
    def wandb_or_print_results(self, result: dict):
        if self.cfg.use_wandb:
            self.wandb.log(result)       
        else:
            print()
            pprint.pprint(result)


    def get_embeddings(self, data_loader, no_loss=False):
        graph_embeddings = []
        labels=[]
        names=[]
        final_loss = 0.00
        with torch.no_grad():
            self.model.eval()
            for batch in data_loader:
                torch.cuda.empty_cache()
                batch.to(self.cfg.device)
                batch.coalesce()
                acfg_graphs = list(itertools.chain.from_iterable(batch.features))
                acfgs = (next(iter(DataLoader(acfg_graphs, batch_size=len(acfg_graphs)))))
                acfgs = acfgs.to(self.cfg.device)
                if 'func' in self.cfg.pcode2vec:
                    graph_embed = self.model(acfgs.features, acfgs.edge_index, acfgs.batch, batch.edge_index, func_pcode_x=batch.func_pcode_features)
                else:
                    graph_embed = self.model(acfgs.features, acfgs.edge_index, acfgs.batch, batch.edge_index)                   
                acfg_labels = torch.LongTensor(list(itertools.chain.from_iterable(batch.function_labels))).to(self.cfg.device)

                test_loss = 0
                non_thunk_func_indexes = torch.where(acfg_labels!=self.thunk_idx)
                graph_embed = graph_embed[non_thunk_func_indexes]
                acfg_labels = acfg_labels[non_thunk_func_indexes]
                graph_embeddings.append(graph_embed)
                labels+=acfg_labels
                temp_names = [] 
                for archi, acfgs_function_names, package in zip(batch.archi, batch.function_names, batch.package):
                    for name in acfgs_function_names: 
                        temp_names.append("%s.%s.%s" % (archi, package, name))
                
                for name_idx, temp_name in enumerate(temp_names):
                    if name_idx in non_thunk_func_indexes[0]:
                        names.append(temp_name)
                
                if not no_loss:
                    func_pair_gt_mtx = (acfg_labels.unsqueeze(0).repeat((graph_embed.size(0),1)) \
                                    == acfg_labels.unsqueeze(1).repeat((1,graph_embed.size(0))))
                    func_pair_gt_mtx = func_pair_gt_mtx.flatten()
                    func_pair_gt_mtx = (func_pair_gt_mtx.float()-0.5)*2

                    # import pdb; pdb.set_trace()
                    all_comb_a = graph_embed.unsqueeze(0).repeat((graph_embed.size(0),1,1)).view(-1,graph_embed.size(1))
                    all_comb_b = graph_embed.unsqueeze(1).repeat((1,graph_embed.size(0),1)).view(-1,graph_embed.size(1)) 
                    
                    all_comb_a = all_comb_a.view(graph_embed.size(0), graph_embed.size(0), graph_embed.size(1))
                    all_comb_b = all_comb_b.view(graph_embed.size(0), graph_embed.size(0), graph_embed.size(1))

                    func_pair_gt_mtx = func_pair_gt_mtx.view(graph_embed.size(0), graph_embed.size(0))
                    # all_comb_a_chunk = torch.chunk(all_comb_a, graph_embed.shape[0])
                    # all_comb_b_chunk = torch.chunk(all_comb_b, graph_embed.shape[0])
                    # import pdb; pdb.set_trace()
                    # gt_mtx_chunk = torch.chunk(func_pair_gt_mtx, graph_embed.shape[0])
                    for a_chunk, b_chunk, gt_chunk in zip(all_comb_a, all_comb_b, func_pair_gt_mtx):
                        test_loss += F.cosine_embedding_loss(a_chunk, b_chunk, gt_chunk, 0.5, reduction='sum')

                    final_loss += test_loss.detach().item()/len(func_pair_gt_mtx)
                
        
        return torch.cat(graph_embeddings, axis=0), torch.tensor(labels).to(self.cfg.device), names, final_loss

    def inference(self, data_loader, eval_set=None, no_loss=False):
        func_embeds, labels, names, loss = self.get_embeddings(data_loader, no_loss)
        if eval_set != None:
            eval_func_embeds, eval_labels, eval_names, _ = self.get_embeddings(eval_set, no_loss)
            start_idx = len(func_embeds)
            func_embeds = torch.cat([func_embeds, eval_func_embeds], axis=0)
            
            labels = torch.cat([labels, eval_labels], axis=0)
            names += eval_names
            eval_hits = [0]*self.cfg.topk
            eval_able_hits = len(func_embeds)-start_idx
        
        hits = [0] * self.cfg.topk
        able_hits = len(func_embeds)
        
        results = defaultdict(dict)

        for func_idx, func_emb in enumerate(func_embeds):
            if "FUN_" in names[func_idx] or ".thunk" in names[func_idx] or\
                "_init" in names[func_idx] or "_fini" in names[func_idx] or\
                "__libc_csu_fini" in names[func_idx] or "__lib_csu_init" in names[func_idx] or\
                "__do_global_dtors_aux" in names[func_idx] or "__gmon_start__" in names[func_idx] or\
                "deregister_tm_clones" in names[func_idx] or "register_tm_clones" in names[func_idx] or\
                "frame_dummy" in names[func_idx] or "_start" in names[func_idx] or\
                "_FINI_0" in names[func_idx] or "_DT_INIT" in names[func_idx]  or\
                "_INIT_0" in names[func_idx] or "entry" in names[func_idx] or "thunk" in names[func_idx]:
            
                ### functions that don't have function name labels shouldn't be evalauted. 
                ### thunk functions also shouldn't be evaluated.
                ### some common functions used in every c programs should not be evaluated too.
                topk_hits = [-1]*self.cfg.topk
            else:
                distances = -F.cosine_similarity(func_emb.unsqueeze(0), func_embeds)
                distances = distances.squeeze()
            
                func_label = labels[func_idx].item()

                new_distances = torch.cat([distances[:func_idx], distances[func_idx+1:]])
                label_set_tensor = torch.cat([labels[:func_idx], labels[func_idx+1:]])
                new_names = names[:func_idx] +  names[func_idx+1:]
                
                if not func_label in label_set_tensor:
                    topk_hits = [-1] * self.cfg.topk

                top_k_values = torch.unique(new_distances, sorted=True)[:self.cfg.topk]

                groups_result = []
                for value in top_k_values:
                    matched_indexes = torch.where(new_distances == value)[0]
                    matched_labels = label_set_tensor[matched_indexes]
                    matched_names = [new_names[idx] for idx in matched_indexes]
                    groups_result.append(matched_labels)
                    results["%d.%s"% (func_idx, names[func_idx])][value.item()] = matched_names

                topk_hits = [0]*self.cfg.topk
                for idx, group in enumerate(groups_result):
                    if func_label in group:
                        topk_hits[idx:self.cfg.topk] = [1]*(self.cfg.topk-idx)
                        break
            if all(hit == -1 for hit in topk_hits):
                if eval_set != None and func_idx >= start_idx:
                    eval_able_hits -= 1
                able_hits -= 1
            else:
                for idx, hit in enumerate(topk_hits):
                    if eval_set != None and func_idx >= start_idx:
                        eval_hits[idx] += hit
                    hits[idx] += hit
        if eval_set != None:
            print([i/eval_able_hits for i in eval_hits])
        return func_embeds, names, [i/able_hits for i in hits], loss, results

    def evaluate(self, eval_set, database, no_loss=False):
        func_embeds_stripped, stripped_labels, stripped_names, loss = self.get_embeddings(eval_set, no_loss)

        func_embeds_database, database_labels, database_names, _ = self.get_embeddings(database, no_loss)
        
        hits = [0] * self.cfg.topk
        able_hits = len(func_embeds_stripped)
        results = defaultdict(dict)
        topk_results = defaultdict(dict)

        for func_idx, func_emb in enumerate(func_embeds_stripped):
            if "FUN_" in stripped_names[func_idx] or ".thunk" in stripped_names[func_idx] or\
                "_init" in stripped_names[func_idx] or "_fini" in stripped_names[func_idx] or\
                "__libc_csu_fini" in stripped_names[func_idx] or "__lib_csu_init" in stripped_names[func_idx] or\
                "__do_global_dtors_aux" in stripped_names[func_idx] or "__gmon_start__" in stripped_names[func_idx] or\
                "deregister_tm_clones" in stripped_names[func_idx] or "register_tm_clones" in stripped_names[func_idx] or\
                "frame_dummy" in stripped_names[func_idx] or "_start" in stripped_names[func_idx] or\
                "_FINI_0" in stripped_names[func_idx] or "_DT_INIT" in stripped_names[func_idx]  or\
                "_INIT_0" in stripped_names[func_idx] or "entry" in stripped_names[func_idx] or "thunk" in stripped_names[func_idx]:
            
                ### functions that don't have function name labels shouldn't be evalauted. 
                ### thunk functions also shouldn't be evaluated.
                ### some common functions used in every c programs should not be evaluated too.
                topk_hits = [-1]*self.cfg.topk
            else:
                distances = -F.cosine_similarity(func_emb.unsqueeze(0), func_embeds_database)
                distances = distances.squeeze()
                #import pdb; pdb.set_trace()
                new_distances = distances
                label_set_tensor = database_labels
                new_names = database_names
                
                func_label = stripped_labels[func_idx].item()

                if not func_label in label_set_tensor:
                    topk_hits = [-1] * self.cfg.topk

                top_k_values = torch.unique(new_distances, sorted=True)[:self.cfg.topk]

                groups_result = []
                for value in top_k_values:
                    matched_indexes = torch.where(new_distances == value)[0]
                    matched_labels = label_set_tensor[matched_indexes]
                    matched_names = [new_names[idx] for idx in matched_indexes]
                    groups_result.append(matched_labels)
                    results["%d.%s"% (func_idx, stripped_names[func_idx])][value.item()] = matched_names   
                k = 0
                scores = list(results["%d.%s"% (func_idx, stripped_names[func_idx])].keys())
                topk_matched_names = []
                for score in scores:
                    matched_names = results["%d.%s"% (func_idx, stripped_names[func_idx])][score]                    
                    for matched_name in matched_names:
                        if k < self.cfg.topk:
                            topk_matched_names.append(matched_name)
                            k += 1
                        else:
                            break
                    if k >= self.cfg.topk:
                        break
                topk_results["%d.%s"% (func_idx, stripped_names[func_idx])] = topk_matched_names

                topk_hits = [0]*self.cfg.topk
                # import pdb; pdb.set_trace()
                '''for idx, name in enumerate(topk_matched_names):
                    #import pdb; pdb.set_trace()
                    if stripped_names[func_idx].split(".")[-1] in name.split(".")[-1]:
                        topk_hits[idx:self.cfg.topk] = [1]*(self.cfg.topk-idx)
                        break'''
                for idx, score in enumerate(scores):
                    matched_names = results["%d.%s"% (func_idx, stripped_names[func_idx])][score]
                    matched_func_names = [name.split(".")[-1] for name in matched_names]
                    if stripped_names[func_idx].split(".")[-1] in matched_func_names:
                        topk_hits[idx:self.cfg.topk] = [1]*(self.cfg.topk-idx)
                        break
                """for idx, group in enumerate(groups_result):
                    if func_label in group:
                        topk_hits[idx:self.cfg.topk] = [1]*(self.cfg.topk-idx)
                        break"""  

            if all(hit == -1 for hit in topk_hits):
                able_hits -= 1
            else:
                for idx, hit in enumerate(topk_hits):
                    hits[idx] += hit   
        # import pdb; pdb.set_trace()
        return [i/able_hits for i in hits], loss, topk_results
        
    def do_func_matching(self, data_loader, eval_set=None, no_loss=False):
        func_embeds_stripped, stripped_labels, stripped_names, loss = self.get_embeddings(data_loader, no_loss)

        func_embeds_unstripped, unstripped_labels, unstripped_names, _ = self.get_embeddings(eval_set, no_loss)
        
        results = defaultdict(dict)

        for func_idx, func_emb in enumerate(func_embeds_stripped):
            distances = -F.cosine_similarity(func_emb.unsqueeze(0), func_embeds_unstripped)
            distances = distances.squeeze()
    
            new_distances = distances
            label_set_tensor = unstripped_labels
            new_names = unstripped_names
            
            top_k_values = torch.unique(new_distances, sorted=True)[:self.cfg.topk]

            groups_result = []
            for value in top_k_values:
                matched_indexes = torch.where(new_distances == value)[0]
                matched_labels = label_set_tensor[matched_indexes]
                matched_names = [new_names[idx] for idx in matched_indexes]
                groups_result.append(matched_labels)
                results["%d.%s"% (func_idx, stripped_names[func_idx])][value.item()] = matched_names        
        return results

    def cluster_inference(self, data_loader, cluster_idx):
        function_embeddings, labels, names, loss = self.get_embeddings(data_loader)
        num_of_functions = sum([len(x.function_names) for x in data_loader.dataset]) 
        label_set_tensor = torch.tensor(labels).to(self.cfg.device)
        hits = [0] * self.cfg.topk
        hits_cluster = [0] * self.cfg.num_clusters
        for i in range(0, self.cfg.num_clusters):
            hits_cluster[i] = [0]*self.cfg.topk
        able_hits = num_of_functions
        able_hits_cluster = Counter(cluster_idx.cpu().tolist())

        for function_idx in tqdm(range(num_of_functions)):
            if names[function_idx].startswith("FUN_") or names[function_idx] == "thunk":
                topk_hits = [-1]*self.cfg.topk
            else:
                distances = -F.cosine_similarity(function_embeddings[function_idx].unsqueeze(dim=0), function_embeddings)
                distances = distances.squeeze()           
            
                label = label_set_tensor[function_idx].item()
                topk_hits, top_k_values, topk_groups_result = get_topk_hits(distances, label_set_tensor, function_idx, label, topk=self.cfg.topk)
                
        
            if all(hit == -1 for hit in topk_hits):
                able_hits_cluster[cluster_idx[function_idx]] -= 1
                able_hits -= 1
            else:
                for idx, hit in enumerate(topk_hits):
                    hits[idx] += hit
                    hits_cluster[cluster_idx[function_idx]][idx] += hit
        #import pdb; pdb.set_trace()
        return function_embeddings, [i/able_hits for i in hits], [[j/able_hits_cluster[i] for j in cluster] for i, cluster in enumerate(hits_cluster)], loss
   
    def save_model(self):
        self.cfg.pml.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(self.cfg.pml)+'/model_weights.pkth')
        config_save_path = self.cfg.pml / "command.txt"
        with open(str(config_save_path), 'w') as config_file:
            config_file.write(" ".join(sys.argv))

    def load_model(self):
        if not self.cfg.pml.exists():
            raise ValueError("Model load path not exist %s" % str(self.cfg.pml))
        self.model.load_state_dict(torch.load(str(self.cfg.pml)+'/model_weights.pkth', map_location=torch.device(self.cfg.device)))
        self.model.to(self.cfg.device)

    def export_cluster_csv(self, cluster, train_hits, test_hits):
        cluster_filename = self.cfg.metrics_path / f"{'cluster_minitest_metrics_.csv'}"
        print("Exporting %s" % cluster_filename)
        results = []
        results += train_hits
        results += test_hits
        pd.DataFrame(data=[results]).to_csv(cluster_filename, mode='a', header=False)