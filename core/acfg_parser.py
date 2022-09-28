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

import copy, os, sys

sys.path.append(os.path.dirname(sys.path[0]))
from collections import defaultdict
from glob import glob
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ACFGDataset:
    def __init__(self, pcode2vec="none"):
        self.func2idx = {}
        self.end2end_dataset = []
        self.pcode2vec = pcode2vec

        self.graphs = {}
        self.must_go_testset_pkg_list = ['ch05','ch06']

    def load_1_binary(self, data_path,num_features):
        contents = self._read_acfg_folder(data_path)
        thunk_func_names = self._get_all_thunk_functions(contents)
        
        tqdm_bar = tqdm(contents)
        for arch_pkg_bin in tqdm_bar:
        
            arch = arch_pkg_bin[0]
            pkg = arch_pkg_bin[1]
            bin = arch_pkg_bin[2]
            
            tqdm_bar.set_description("Processing %s, %s, %s" % (arch, pkg, bin))
            
            for archi_dir_path in contents[arch_pkg_bin]:
                bin_cg = nx.DiGraph()

                call_graph_edges = self._get_callgraph_edges(archi_dir_path/"callgraph.csv")
                #print("archi_dir_path: " + str(archi_dir_path))
                block2features, func2blocks = self._get_blockfeatures(archi_dir_path/"attributes.csv")
                block2blocks = self._get_acfg_edges(archi_dir_path/"edges.csv")
                
                if self.pcode2vec == 'bb':
                    block2features_pcode, missing_basic_blocks = self._get_pcodefeatures(archi_dir_path/"bb_pcode_attribute.csv", block2features)               
                    block2features = block2features_pcode
                    # missing basic blocks are the blocks in bb_pcode_attribute.csv but not in attribute.csv. 
                    self._error_handling_missing_blocks(missing_basic_blocks, func2blocks, block2features, block2blocks)
                
                elif self.pcode2vec == 'func':
                    func2features_pcode = self._get_func_pcodefeatures(archi_dir_path/"pcode_attribute.csv")
                    pass
                
                elif self.pcode2vec == 'bb_func':
                    block2features_pcode, missing_basic_blocks = self._get_pcodefeatures(archi_dir_path/"bb_pcode_attribute.csv", block2features)               
                    block2features = block2features_pcode
                    # missing basic blocks are the blocks in bb_pcode_attribute.csv but not in attribute.csv. 
                    self._error_handling_missing_blocks(missing_basic_blocks, func2blocks, block2features, block2blocks)
                    func2features_pcode = self._get_func_pcodefeatures(archi_dir_path/"pcode_attribute.csv")
                    pass

                else:
                    pass

                for func, blocks in func2blocks.items():
                    func_cfg = nx.DiGraph()
                    for block in blocks:
                        block_index = "%s.%s" % (func, block)
                        if block_index in block2features:
                            func_cfg.add_node(block_index, features=block2features[block_index])
                            for neighbor in block2blocks[block_index]:
                                func_cfg.add_edge(block_index, neighbor)

                    if 'func' in self.pcode2vec:
                        try: 
                            bin_cg.add_node(func, features=func_cfg, type="normal", func_pcode_features=func2features_pcode[func])
                        except KeyError:
                            import pdb; pdb.set_trace()
                    else:
                        bin_cg.add_node(func, features=func_cfg, type="normal")

                feature_dim_bb = num_features+12 if 'bb' in self.pcode2vec else 12
                feature_dim_func = num_features if 'func' in self.pcode2vec else 0
                
                for from_edge, to_edge in call_graph_edges:
                    if not from_edge in bin_cg.nodes():
                        if from_edge in thunk_func_names[(arch, pkg, bin)]:
                            if 'func' in self.pcode2vec:
                                bin_cg.add_node(from_edge, features=nx.DiGraph(), type="normal", func_pcode_features=[0.0]*feature_dim_func) 
                            else:
                                bin_cg.add_node(from_edge, features=nx.DiGraph(), type="normal")
                            bin_cg.nodes()[from_edge]['type'] = "thunk"
                            bin_cg.nodes()[from_edge]['features'].add_node("dummy", features=[0.0]*feature_dim_bb)

                    if not to_edge in bin_cg.nodes():
                        if to_edge in thunk_func_names[(arch, pkg, bin)]:
                            if 'func' in self.pcode2vec:
                                bin_cg.add_node(to_edge, features=nx.DiGraph(), type="normal", func_pcode_features=[0.0]*feature_dim_func)
                            else:
                                bin_cg.add_node(to_edge, features=nx.DiGraph(), type="normal")
                            bin_cg.nodes()[to_edge]['type'] = "thunk"
                            bin_cg.nodes()[to_edge]['features'].add_node("dummy", features=[0.0]*feature_dim_bb)
                            
                    if from_edge in bin_cg.nodes() and to_edge in bin_cg.nodes():
                        bin_cg.add_edge(from_edge, to_edge)

                #### the process that filters the FUN_XXX functions. 
                # call_graph.remove_nodes_from([x for x in call_graph.nodes() if "FUN_" in x])
                bin_cg.store_path = archi_dir_path
                # we should append func features roughly here to each bin_cg
                if arch not in self.graphs:
                    self.graphs[arch]={}
                if pkg not in self.graphs[arch]:
                    self.graphs[arch][pkg] = {}
                self.graphs[arch][pkg][bin] = bin_cg
                 
        
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                func_dict_in_pkg = {}
                for bin, cg in bins.items():
                    # import pdb; pdb.set_trace()
                    for func, func_attr in cg.nodes(data=True):
                        if func_attr['type'] != "thunk": 
                            func_dict_in_pkg[func] = func_attr
                # import pdb; pdb.set_trace() 
                for bin, cg in bins.items():    
                    for func, func_attr in cg.nodes(data=True):
                        if func_attr['type'] == 'thunk' and func in func_dict_in_pkg:
                            func_attr["features"] = func_dict_in_pkg[func]['features']
                            if "func_pcode_features" in func_dict_in_pkg[func]:
                                func_attr["func_pcode_features"] = func_dict_in_pkg[func]['func_pcode_features']
                            func_attr["type"] = "external"      
        
        # removed isolated nodes.
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    cg.remove_nodes_from(list(nx.isolates(cg)))
      
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    thunk_func = [func_name for func_name, func_attr in cg.nodes(data=True) if func_attr['type'] == "thunk"]
                    cg.remove_nodes_from(thunk_func)

        no_arch_func_names = set()
        for pkgs in self.graphs.values():
            for bins in pkgs.values():
                for cg in bins.values():
                    for func_name, func_attr in cg.nodes(data=True):
                        if func_attr['type'] != "thunk":
                            no_arch_func_names.add(func_name)

        no_arch_func_names.add("thunk")

        self.func2idx = {v:k for k, v in enumerate(list(no_arch_func_names))}
        self.idx2func = {v:k for k, v in self.func2idx.items()}

    def load(self, dataset_path, num_features, vis_only=False):
        contents = self._read_acfg_folders(dataset_path)
        thunk_func_names = self._get_all_thunk_functions(contents)
        
        tqdm_bar = tqdm(contents)
        for arch_pkg_bin in tqdm_bar:
        
            arch = arch_pkg_bin[0]
            pkg = arch_pkg_bin[1]
            bin = arch_pkg_bin[2]
            
            tqdm_bar.set_description("Processing %s, %s, %s" % (arch, pkg, bin))
            
            for archi_dir_path in contents[arch_pkg_bin]:
                bin_cg = nx.DiGraph()

                call_graph_edges = self._get_callgraph_edges(archi_dir_path/"callgraph.csv")
                #print("archi_dir_path: " + str(archi_dir_path))
                block2features, func2blocks = self._get_blockfeatures(archi_dir_path/"attributes.csv")
                block2blocks = self._get_acfg_edges(archi_dir_path/"edges.csv")
                
                if self.pcode2vec == 'bb':
                    block2features_pcode, missing_basic_blocks = self._get_pcodefeatures(archi_dir_path/"bb_pcode_attribute.csv", block2features)               
                    block2features = block2features_pcode
                    # missing basic blocks are the blocks in bb_pcode_attribute.csv but not in attribute.csv. 
                    try:
                        self._error_handling_missing_blocks(missing_basic_blocks, func2blocks, block2features, block2blocks)
                    except:
                        continue
                
                elif self.pcode2vec == 'func':
                    func2features_pcode = self._get_func_pcodefeatures(archi_dir_path/"pcode_attribute.csv")
                    pass
                
                elif self.pcode2vec == 'bb_func':
                    block2features_pcode, missing_basic_blocks = self._get_pcodefeatures(archi_dir_path/"bb_pcode_attribute.csv", block2features)               
                    block2features = block2features_pcode
                    # missing basic blocks are the blocks in bb_pcode_attribute.csv but not in attribute.csv. 
                    self._error_handling_missing_blocks(missing_basic_blocks, func2blocks, block2features, block2blocks)
                    func2features_pcode = self._get_func_pcodefeatures(archi_dir_path/"pcode_attribute.csv")
                    pass

                else:
                    pass

                for func, blocks in func2blocks.items():
                    func_cfg = nx.DiGraph()
                    for block in blocks:
                        block_index = "%s.%s" % (func, block)
                        if block_index in block2features:
                            func_cfg.add_node(block_index, features=block2features[block_index])
                            for neighbor in block2blocks[block_index]:
                                func_cfg.add_edge(block_index, neighbor)

                    if 'func' in self.pcode2vec:
                        try: 
                            bin_cg.add_node(func, features=func_cfg, type="normal", func_pcode_features=func2features_pcode[func])
                        except KeyError:
                            import pdb; pdb.set_trace()
                    else:
                        bin_cg.add_node(func, features=func_cfg, type="normal")

                feature_dim_bb = num_features+12 if 'bb' in self.pcode2vec else 12
                feature_dim_func = num_features if 'func' in self.pcode2vec else 0
                
                for from_edge, to_edge in call_graph_edges:
                    if not from_edge in bin_cg.nodes():
                        if from_edge in thunk_func_names[(arch, pkg, bin)]:
                            if 'func' in self.pcode2vec:
                                bin_cg.add_node(from_edge, features=nx.DiGraph(), type="normal", func_pcode_features=[0.0]*feature_dim_func) 
                            else:
                                bin_cg.add_node(from_edge, features=nx.DiGraph(), type="normal")
                            bin_cg.nodes()[from_edge]['type'] = "thunk"
                            bin_cg.nodes()[from_edge]['features'].add_node("dummy", features=[0.0]*feature_dim_bb)

                    if not to_edge in bin_cg.nodes():
                        if to_edge in thunk_func_names[(arch, pkg, bin)]:
                            if 'func' in self.pcode2vec:
                                bin_cg.add_node(to_edge, features=nx.DiGraph(), type="normal", func_pcode_features=[0.0]*feature_dim_func)
                            else:
                                bin_cg.add_node(to_edge, features=nx.DiGraph(), type="normal")
                            bin_cg.nodes()[to_edge]['type'] = "thunk"
                            bin_cg.nodes()[to_edge]['features'].add_node("dummy", features=[0.0]*feature_dim_bb)
                            
                    if from_edge in bin_cg.nodes() and to_edge in bin_cg.nodes():
                        bin_cg.add_edge(from_edge, to_edge)

                #### the process that filters the FUN_XXX functions. 
                # call_graph.remove_nodes_from([x for x in call_graph.nodes() if "FUN_" in x])
                bin_cg.store_path = archi_dir_path
                # we should append func features roughly here to each bin_cg
                if arch not in self.graphs:
                    self.graphs[arch]={}
                if pkg not in self.graphs[arch]:
                    self.graphs[arch][pkg] = {}
                self.graphs[arch][pkg][bin] = bin_cg
                 
        
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                func_dict_in_pkg = {}
                for bin, cg in bins.items():
                    # import pdb; pdb.set_trace()
                    for func, func_attr in cg.nodes(data=True):
                        if func_attr['type'] != "thunk": 
                            func_dict_in_pkg[func] = func_attr
                # import pdb; pdb.set_trace() 
                for bin, cg in bins.items():    
                    for func, func_attr in cg.nodes(data=True):
                        if func_attr['type'] == 'thunk' and func in func_dict_in_pkg:
                            func_attr["features"] = func_dict_in_pkg[func]['features']
                            if "func_pcode_features" in func_dict_in_pkg[func]:
                                func_attr["func_pcode_features"] = func_dict_in_pkg[func]['func_pcode_features']
                            func_attr["type"] = "external"      
        
        # removed isolated nodes.
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    cg.remove_nodes_from(list(nx.isolates(cg)))

        # "Visualizing Packing Dataset"
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    # print("Visualizing %s.%s.%s" % (arch, pkg, bin))
                    for func_name, func_attr in cg.nodes(data=True):
                        if func_attr['type'] == "thunk":
                            func_attr['fontcolor'] = "red"
                            func_attr['color'] = "red"
                        elif func_attr['type'] == "external":
                            func_attr['fontcolor'] = "blue"
                            func_attr['color'] = "blue"
                        else:
                            func_attr['fontcolor'] = "black"
                            func_attr['color'] = "black"
                    # parent_dirpath = (cg.store_path/"..").resolve()              
                    # nx.drawing.nx_pydot.write_dot(cg, str(parent_dirpath) + "/%s.dot" % parent_dirpath.name)
                    # for func_name, func_attr in cg.nodes(data=True):
                    #     func_cfg = func_attr['features']
                    #     parent_folder_path = (cg.store_path/"..").resolve()
                    #     ccode_path = parent_folder_path / (parent_folder_path.name + "-ccode")
                    #     nx.drawing.nx_pydot.write_dot(func_cfg, str(ccode_path) + "/%s.dot" % func_name)
        
        if vis_only == True:
            return 

        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    thunk_func = [func_name for func_name, func_attr in cg.nodes(data=True) if func_attr['type'] == "thunk"]
                    cg.remove_nodes_from(thunk_func)

        no_arch_func_names = set()
        for pkgs in self.graphs.values():
            for bins in pkgs.values():
                for cg in bins.values():
                    for func_name, func_attr in cg.nodes(data=True):
                        if func_attr['type'] != "thunk":
                            no_arch_func_names.add(func_name)

        '''for pkgs in self.graphs.values():
            for bins in pkgs.values():
                for cg in bins.values():'''
                    #import pdb; pdb.set_trace()

        no_arch_func_names.add("thunk")

        self.func2idx = {v:k for k, v in enumerate(list(no_arch_func_names))}
        self.idx2func = {v:k for k, v in self.func2idx.items()}

    
    def pack4cfg2vec(self):
        from torch_geometric.utils.convert import from_networkx
        self.end2end_dataset = []
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    data_obj = from_networkx(cg)
                    features = []
                    for feature in data_obj.features:
                        data = from_networkx(feature)
                        data.edge_index = data.edge_index.long()
                        try:
                            data.features = data.features.float()
                        except:
                            print(pkg, bin, arch)
                        features.append(data)
                    data_obj.features =  features
                    # data_obj.func_features = xxx
                    data_obj.archi = arch
                    data_obj.package = "%s.%s" % (pkg, bin)
                    function_names = []
                    function_labels = []
                    for func_name, func_attr in cg.nodes(data=True):
                        if func_attr['type'] == "thunk":
                            function_names.append("thunk")
                            function_labels.append(self.func2idx["thunk"])
                        else:
                            function_names.append(func_name)
                            function_labels.append(self.func2idx[func_name])
                    data_obj.function_names = function_names
                    data_obj.function_labels = function_labels
                    self.end2end_dataset.append(data_obj)

    def pack4gemini(self):
        import torch
        from core.utils import read_graph_allstar
        func2graphs = {}
        for arch, pkgs in self.graphs.items():
            for pkg, bins in pkgs.items():
                for bin, cg in bins.items():
                    for func_name, func_attr in cg.nodes(data=True):
                        func_acfg = func_attr["features"]
                        
                        node_info = dict(func_acfg.nodes("features"))
                        node_names = list(node_info.keys())
                        features = torch.from_numpy(np.vstack(node_info.values())).float()

                        source_edges = []
                        target_edges = []
                        for edge in func_acfg.edges:
                            source_edges.append(node_names.index(edge[0]))
                            target_edges.append(node_names.index(edge[1]))
                        
                        # the shape of A is expected to be [2, num_edges]
                        if len(source_edges) > 0:
                            A = torch.transpose(torch.LongTensor(list(zip(source_edges, target_edges))), 0, 1) # creating A (initial node adjancency information)
                        else:
                            A = torch.empty(size=[2,0], dtype=torch.long)

                        if func_attr['type'] == "thunk":
                            label = self.func2idx["thunk"]
                        else:
                            label = self.func2idx[func_name]
                        func2graphs[(arch, pkg, bin, func_name)] = (features, A, node_names, label)
        print("Converting Allstar data to Gemini format")
        self.Gs, self.classes, self.packages = read_graph_allstar(func2graphs, self.func2idx)

    def split_dataset_by_package(self, test_size, seed):
        ## add-in must go testset pkg list.
        package_list = np.unique([x.package.split('.')[0] for x in self.end2end_dataset if x.package.split('.')[0] not in self.must_go_testset_pkg_list]).tolist()       
        np.random.seed(seed)
        perm=np.random.permutation(len(package_list))
        train_package_set = []
        test_package_set = []
        for pkg_idx in perm[0:int(len(package_list)*(1-test_size))]:
            train_package_set.append(package_list[pkg_idx])
        for pkg_idx in perm[int(len(package_list)*(1-test_size)):]:
            test_package_set.append(package_list[pkg_idx])
        # train_package_set, test_package_set = train_test_split(package_list, test_size=test_size, \
        #                                                        shuffle=True, random_state=seed)
        train_set = [] 
        test_set = []
        eval_set = []
        for data_point in self.end2end_dataset:
            pkg_name = data_point.package.split('.')[0]
            if pkg_name in train_package_set:
                train_set.append(data_point)
            elif pkg_name in test_package_set:
                test_set.append(data_point)
            elif pkg_name in self.must_go_testset_pkg_list:
                eval_set.append(data_point)

        print("Spliting dataset into Training/Testing/Eval sets (%d, %d, %d)" % (len(train_set), len(test_set), len(eval_set)))
        return train_set, test_set, eval_set
    
    
    def split_dataset_by_binary(self, test_size, seed):
        ## add-in must go testset pkg list.
        package_list = np.unique([x.package for x in self.end2end_dataset]).tolist()
        train_package_set, test_package_set = train_test_split(package_list, test_size=test_size, \
                                                               shuffle=True, random_state=seed)
        train_set = [] 
        test_set = []
        for data_point in self.end2end_dataset:
            if data_point.package in train_package_set:
                train_set.append(data_point)
            elif data_point.package in test_package_set:
                test_set.append(data_point)

        print("Spliting dataset into Training/Testing sets (%d, %d)" % (len(train_set), len(test_set)))
        return train_set, test_set
    
    def _get_all_thunk_functions(self, contents):
        thunk_func_names = defaultdict(list)
        for arch_pkg_bin in contents:
            for arch_dir_path in contents[arch_pkg_bin]:
                parent_folder_path = (arch_dir_path/"..").resolve()
                ccode_path = parent_folder_path / (parent_folder_path.name + "-ccode")
                for code_text_path in glob("%s/**" % ccode_path):
                    if "_thunk" ==  Path(code_text_path).stem[-6:]:
                        thunk_func_names[arch_pkg_bin].append(Path(code_text_path).stem[:-6])
        
        # ex:  'i386.adns-tools.libadns.so.1.5': ['__assert_fail',
        #                                         'realloc',
        #                                         '_ITM_registerTMCloneTable']

        return thunk_func_names
    
    def _read_acfg_folders(self, dataset_path):
        ''' acfg_folders is structured using package name as the key a list of containing acfg folders as content'''
        acfg_folders = defaultdict(list)

        for folder_1st_level in glob("%s/**" % str(dataset_path)):
            for folder_2nd_level in glob("%s/**" % str(folder_1st_level)):
                acfg_folder_name = Path(folder_2nd_level).name
                if acfg_folder_name.endswith("acfg"):
                    package_binary = "-".join(acfg_folder_name.split("-")[:-2])
                    archi = acfg_folder_name.split("-")[-2].split(".")[0]
                    package = package_binary.split("___")[0]
                    bin = package_binary.split("___")[1]
                    acfg_folders[(archi,package,bin)].append(Path(folder_2nd_level))
        return acfg_folders

    def _read_acfg_folder(self, dataset_path):
        ''' acfg_folders is structured using package name as the key a list of containing acfg folders as content'''
        acfg_folders = defaultdict(list)        
        for folder_2nd_level in glob("%s/**" % str(dataset_path)):
            acfg_folder_name = Path(folder_2nd_level).name
            if acfg_folder_name.endswith("acfg"):
                package_binary = "-".join(acfg_folder_name.split("-")[:-2])
                archi = acfg_folder_name.split("-")[-2].split(".")[0]
                package = package_binary.split("___")[0]
                bin = package_binary.split("___")[1]
                acfg_folders[(archi,package,bin)].append(Path(folder_2nd_level))
        return acfg_folders
              
    def _get_callgraph_edges(self, callgraph_edge_file_path):
        call_graph_edges = list()
        with open(str(callgraph_edge_file_path), "r") as fin:
            for line in fin.readlines():
                content = line.strip().split(", ")
                if (not (content[1], content[2]) in call_graph_edges):
                    call_graph_edges.append((content[1], content[2]))
        return call_graph_edges

    def _get_blockfeatures(self, block2feature_file_path):
        block2features = {}
        func2blocks = defaultdict(list)
       
        with open(str(block2feature_file_path), "r") as fin:          
            for line in fin.readlines():
                if line.strip() != "":
                    data_row = line.strip().split(",")
                    func_name = data_row[0] # currently not used.
                    block_name = data_row[1]
                    block_feature = data_row[2:]
                    block_index = "%s.%s" % (func_name, block_name)
                    features = np.array(list(map(int, block_feature)))
                    block2features[block_index] = features
                    func2blocks[func_name].append(block_name)
        return block2features, func2blocks

    def _get_pcodefeatures(self, pcode_file_path, block2features):
        
        curr_block2features = copy.deepcopy(block2features)
        with open(str(pcode_file_path), "r") as fin:
            for line in fin.readlines():
                func_name = line.strip().split(",")[0]
                block_name = line.strip().split(", ")[1]
                block_index = "%s.%s" % (func_name, block_name)

                if block_name == "" or curr_block2features[block_index].size>12: 
                    # there can be some lines without block names.
                    continue

                block_pcode_feature = line.strip().split(", ")[2:]
                features = np.array(list(map(float, block_pcode_feature)))
                curr_block2features[block_index] = np.concatenate([curr_block2features[block_index].astype(np.float32), features])
        
        missing_basic_blocks = []
        for block_name, block_feature in curr_block2features.items():
            try:
                if block_feature.size == 12:
                    missing_basic_blocks.append(block_name)
                    print(pcode_file_path, block_name, block_feature)
            except ValueError:
                import pdb;pdb.set_trace()
        
        return curr_block2features, missing_basic_blocks

    def _get_func_pcodefeatures(self, pcode_file_path):
        
        block2features = {}
        with open(str(pcode_file_path), "r") as fin:
            for line in fin.readlines():
                
                func_name = line.strip().split(",")[0]
                func_pcode_feature = line.strip().split(",")[1:]
                features = np.array(list(map(float, func_pcode_feature)))
                block2features[func_name] = features
        
        return block2features

    def _get_acfg_edges(self, edge_file_path):
        block2blocks = defaultdict(list)
        with open(str(edge_file_path), "r") as fin:
            for line in fin.readlines():
                if line.strip():
                    func_name = line.strip().split(",")[0]
                    from_block, to_block = line.strip().split(",")[1:]
                    from_block_index = "%s.%s" % (func_name.strip(), from_block.strip())
                    to_block_index = "%s.%s" % (func_name.strip(), to_block.strip())
                    block2blocks[from_block_index].append(to_block_index)
        return block2blocks

    def _error_handling_missing_blocks(self, missing_bbs, func2blocks, block2features, block2blocks):
        for neighbor_list in block2blocks.values():
            for missed_basic_block in missing_bbs:
                if missed_basic_block in neighbor_list:
                    neighbor_list.remove(missed_basic_block)
                
        for blocks in func2blocks.values():
            for missed_basic_block in missing_bbs:
                if missed_basic_block in blocks:
                    blocks.remove(missed_basic_block)

        for missed_basic_block in missing_bbs:
            del block2blocks[missed_basic_block]

        for missed_basic_block in missing_bbs:
            del block2features[missed_basic_block]

    def __str__(self):
        num_acfgs = sum([len(acg.function_names) for acg in self.end2end_dataset])
        all_acfg_nodes = 0
        all_acfg_edges = 0

        for package in self.end2end_dataset:
            for function in package.features:
                all_acfg_nodes += function.features.shape[0]
                all_acfg_edges += function.edge_index.shape[1]
        
        results=[]
        results.append("number of ACFGs: {}".format(num_acfgs))
        results.append("number of packages/ACGs: {}".format(len(self.end2end_dataset)))
        results.append("average number of edges in an ACFG: {0:f}".format(all_acfg_edges / num_acfgs))
        results.append("average number of nodes in an ACFG: {0:f}".format(all_acfg_nodes / num_acfgs))
        results.append("number of unique labels in dataset: {}".format(len(self.func2idx.values())))

        return "\n".join(results)

    def vis_distribution(self, metrics_path):
        metrics_path.mkdir(parents=True, exist_ok=True)
        num_acfgs = sum([len(acg.function_names) for acg in self.end2end_dataset])
        acg_num_nodes = []
        acg_num_edges = []
        acfg_num_nodes = []
        acfg_num_edges = []
        acfg_num_instructions = []
        for package in self.end2end_dataset:
            acg_num_nodes.append(len(package.function_labels))
            acg_num_edges.append(package.edge_index.shape[1])
            for function in package.features:
                num_instructions = []
                acfg_num_nodes.append(function.features.shape[0])
                acfg_num_edges.append(function.edge_index.shape[1])
                for i in range (0, len(function.features[0])):
                    num_instructions.append(int(function.features[0][i].item()))
                acfg_num_instructions.append(num_instructions)
        
        acg_df = pd.DataFrame(columns=["acg nodes", "acg edges"])
        acg_df["acg nodes"] = acg_num_nodes
        acg_df["acg edges"] = acg_num_edges
        acg_df.to_csv(metrics_path / ('allstar-%d-acg-distributions.csv'% num_acfgs))

        acfg_df = pd.DataFrame(columns=["acfg nodes", "acfg edges"])
        acfg_df["acfg nodes"] = acfg_num_nodes
        acfg_df["acfg edges"] = acfg_num_edges
        acfg_df.to_csv(metrics_path / ('allstar-%d-acfg-distributions.csv'% num_acfgs))

        acfg_instr_df_filename = metrics_path / ('allstar-%d-acfg-instr-distributions.csv'% num_acfgs)
        acfg_instr_df = pd.DataFrame(columns=["total", "arithmetic", "logic", "transfer", "call", "datatransfer", "ssa", "compare", "pointer", "other", "totalConstants", "totalStrings"])
        acfg_instr_df.to_csv(acfg_instr_df_filename)
        acfg_instr_df["acfg nodes"] = acfg_num_nodes
        acfg_instr_df["acfg edges"] = acfg_num_edges
        for function in acfg_num_instructions:
            acfg_instr_df = pd.DataFrame(data=[function]).to_csv(acfg_instr_df_filename, mode='a', header=False)
