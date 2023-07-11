import numpy as np
import pandas as pd
import collections



join_types = ['semi', 'inner', 'anti', 'full', 'right', 'left']

parent_rel_types = ['inner', 'outer', 'subquery', 'member']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed', 'mixed']

setop_coms = ['Except', 'Intersect']



class featurizer():
    
    def __init__(self, REL_NAMES, REL_ATTR_LIST_DICT, index_names, col_min_max, ds_info):

        self.REL_NAMES = REL_NAMES
        self.REL_ATTR_LIST_DICT = REL_ATTR_LIST_DICT
        self.index_names = index_names
        self.col_min_max = col_min_max
        self.ds_info = ds_info
        
        self.max_num_attr = max([len(i) for i in self.REL_ATTR_LIST_DICT.values()])
        self.num_rel = len(self.REL_NAMES)
        self.num_index = len(self.index_names)
                
        self.input_func_dict =         {
            "Hash Join": self.get_join_input,
            "Merge Join": self.get_join_input,
            "Seq Scan": self.get_scan_input,
            "Index Scan": self.get_index_scan_input,
            "Index Only Scan": self.get_index_scan_input,
            "Bitmap Heap Scan": self.get_scan_input,
            "Bitmap Index Scan": self.get_bitmap_index_scan_input,
            "Subquery Scan": self.get_scan_input,
            "Sort": self.get_sort_input,
            "Hash": self.get_hash_input,
            "Aggregate": self.get_aggreg_input,
            'CTE Scan': self.get_scan_input,
            'Incremental Sort': self.get_sort_input,
            'Merge Append': self.get_sort_input,
            'SetOp': self.get_SetOp_input,

            'WindowAgg': self.get_basics,
            'Append': self.get_basics,
            'Group': self.get_basics,
            'Unique': self.get_basics,
            'Result': self.get_basics,
            'BitmapAnd': self.get_basics
        }

        self.input_func = collections.defaultdict(lambda: self.get_basics, self.input_func_dict)
        
        
        
    def get_basics(self, root):
        return [self.ds_info.cost_norm.normalize_label(root.width), self.ds_info.cost_norm.normalize_label(root.card_est), self.ds_info.cost_norm.normalize_label(root.cost_est)]

    def get_rel_one_hot(self, rel_name):
        arr = [0] * self.num_rel
        if rel_name:
            arr[self.REL_NAMES.index(rel_name)] = 1
        return arr

    def get_index_one_hot(self, index_name):
        arr = [0] * self.num_index
        if index_name and index_name in self.index_names:
            arr[self.index_names.index(index_name)] = 1
        return arr


    def get_rel_attr_one_hot(self, rel_name, filter_line):
        attr_list = self.REL_ATTR_LIST_DICT[rel_name]

        min_vec, max_vec = [0] * self.max_num_attr, [0] * self.max_num_attr

        for idx, attr in enumerate(attr_list):
            if attr in filter_line:
                min_vec[idx] = self.col_min_max[attr][0]
                max_vec[idx] = self.col_min_max[attr][1]
        return min_vec + max_vec    
    
    
    def get_scan_input(self, root):
        # root: dict where the root['node_type'] = 'Seq Scan'
        rel_vec = self.get_rel_one_hot(root.table)
#         print('rel_vec: ', rel_vec)
#         print('self.get_basics(root): ', self.get_basics(root))

        try:
            rel_attr_vec = self.get_rel_attr_one_hot(root.table, root.filters)
        except:
            rel_attr_vec = [0] * self.max_num_attr * 2
#         print('rel_attr_vec: ', rel_attr_vec)
        return self.get_basics(root) + rel_vec + rel_attr_vec


    def get_index_scan_input(self, root):
        # plan_dict: dict where the plan_dict['node_type'] = 'Index Scan'

        rel_vec = self.get_rel_one_hot(root.table)
        index_vec = self.get_index_one_hot(root.index)

        try:
            rel_attr_vec = self.get_rel_attr_one_hot(root.table, root.filters)
        except:
    #         if 'Index Cond' in plan_dict:
    #             print('********************* default rel_attr_vec *********************')
    #             print(plan_dict)
            rel_attr_vec = [0] * self.max_num_attr * 2

        res = self.get_basics(root) + rel_vec + rel_attr_vec + index_vec
        return res


    def get_bitmap_index_scan_input(self, root):
        # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'
        index_vec = self.get_index_one_hot(root.index)

        return self.get_scan_input(root) + index_vec

    def get_hash_input(self, root):
        return self.get_basics(root) + [root.hash]

    def get_join_input(self, root):
        type_vec = [0] * len(join_types)
    #     print(root.join_type)
        type_vec[join_types.index(root.join_type)] = 1
        par_rel_vec = [0] * len(parent_rel_types)
        if root.parent_rel:
            par_rel_vec[parent_rel_types.index(root.parent_rel)] = 1
        return self.get_basics(root) + type_vec + par_rel_vec

    def get_sort_key_input(self, root):
        kys = root.sort_key
        one_hot = [0] * (self.num_rel * self.max_num_attr)
        for key in kys:
            key = key.replace('(', ' ').replace(')', ' ')
            for subkey in key.split(" "):
                if subkey != ' ' and '.' in subkey:
                    rel_name, attr_name = subkey.split(' ')[0].split('.')
                    if attr_name[-1] ==  ',':
                        attr_name = attr_name[:-1]
                    if rel_name in self.REL_NAMES:
#                         print(rel_name)
                        one_hot[self.REL_NAMES.index(rel_name) * self.max_num_attr
                                + self.REL_ATTR_LIST_DICT[rel_name].index(attr_name.lower())] = 1

        return one_hot

    def get_sort_input(self, root):
        sort_meth = [0] * len(sort_algos)
        if root.sort_method and ("external" not in root.sort_method):
            sort_meth[sort_algos.index(root.sort_method)] = 1

        return self.get_basics(root) + self.get_sort_key_input(root) + sort_meth

    def get_aggreg_input(self, root):
        strat_vec = [0] * len(aggreg_strats)
        strat_vec[aggreg_strats.index(root.strategy)] = 1
        partial_mode_vec = [0] if root.para_aware == 'false' else [1]
        return self.get_basics(root) + strat_vec + partial_mode_vec


    def get_SetOp_input(self, root):
        setop_com = [0] * len(setop_coms)
        if root.command:
            setop_com[setop_coms.index(root.command)] = 1
        return self.get_basics(root) + setop_com

    def featurize(self, root):
#         print('input_func: ',self.input_func[root.nodeType])
        return self.input_func[root.nodeType](root)






