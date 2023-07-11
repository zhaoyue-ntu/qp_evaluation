from torch import FloatTensor, nn, cat
import copy
import re
import numpy as np
import time




JOIN_TYPES = ['Hash Join', 'Nested Loop', 'Merge Join', 'Other Join']
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan", 
              'Bitmap Heap Scan', 'Subquery Scan']
# ALL_TYPES = JOIN_TYPES + len(REL_NAMES) * SCAN_TYPES


class Batch():
    def __init__(self, flat_trees, q_vecs, indexes):
        self.trees = flat_trees
        self.q_vecs = q_vecs
        self.idxes = indexes
    def to(self,dev):
        self.trees = self.trees.to(dev)
        self.q_vecs = self.q_vecs.to(dev)
        self.idxes = self.idxes.to(dev)
        return self



# +
from TreeConvolution.util import *
def prepare_trees(trees, transformer, left_child, right_child, cuda=False):
#     start_time_padtree = time.time()
    flat_trees = [flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)
#     print("--- tree padcombine takes %s seconds ---" % (time.time() - start_time_padtree))
    
#     start_time_padind = time.time()
    indexes = [tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()
#     print("--- index padcombine takes %s seconds ---" % (time.time() - start_time_padind))

    return (flat_trees, indexes)


# -

def get_plan_stats(roots):
    rows = []
    
    def recurse(n):
        rows.append(n.card_est)
        if n.children != []:
            for child in n.children:
                recurse(child)

    for root in roots:
        recurse(root)
    rows = np.array(rows)
    rows = np.log(rows + 1)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    return rows_min, rows_max


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def collate(x):
    q_vecs = []
    trees = []
    targets = []
  
    for tree, target in x:
        tr, q_vec = tree
        trees.append(tr)
        q_vecs.append(q_vec)
        targets.append(target)
    q_vecs = torch.from_numpy(np.asarray(q_vecs)).float()
    targets = torch.FloatTensor(targets).reshape(-1, 1)
#     start_time_prepare = time.time()
    flat_trees, indexes = prepare_trees(trees, features, left_child, right_child)
#     print("--- prepare tree takes %s seconds ---" % (time.time() - start_time_prepare))
    
    return Batch(flat_trees, q_vecs, indexes), targets

# +
# def collate(x):
#     q_vecs = []
#     trees = []
#     targets = []
  
#     for tree, target in x:
#         tr, q_vec = tree
#         trees.append(tr)
#         q_vecs.append(q_vec)
#         targets.append(target)

#     targets = torch.FloatTensor(targets).reshape(-1, 1)
# #     start_time_prepare = time.time()
#     flat_trees, indexes = prepare_trees(trees, features, left_child, right_child)
# #     print("--- prepare tree takes %s seconds ---" % (time.time() - start_time_prepare))
    
#     #concate here
# #     start_time_concat = time.time()
#     assert len(flat_trees) == len(q_vecs)
#     aug_trees = FloatTensor([[[0]*(len(flat_trees[0][0])+len(q_vecs[0]))]*len(flat_trees[0])]*len(flat_trees))
#     for i in range(len(flat_trees)):
#         for j in range(len(flat_trees[i])):
#             tr = cat((flat_trees[i][j], q_vecs[i]))
#             aug_trees[i][j] = tr
# #     print("--- collation concate takes %s seconds ---" % (time.time() - start_time_concat))
#     aug_trees = aug_trees.transpose(1, 2)

    
#     return Batch(aug_trees, indexes), targets
# -




def is_join(node):
    return len(node.children) >= 2


def is_scan(node):
    if node.children == []:
        return True
    return node.nodeType in SCAN_TYPES




class TreeBuilder:

    def __init__(self, relations, rel_attr_list_dict, alias2table, rows_min, rows_max):
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)
        self.REL_NAMES = self.__relations
        self.NUM_REL = len(self.__relations)
        self.ALL_TYPES = JOIN_TYPES + self.NUM_REL * SCAN_TYPES
        self.REL_ATTR_LIST_DICT = rel_attr_list_dict
        self.alias2table = alias2table
        self.rows_min = rows_min
        self.rows_max = rows_max

    ################################################ query level encoding ################################################

    def __featurize_qjoin(self, node, rel_dict, sub_query_rels, alias2table):     
        '''
        encode the upper tri of join matrix by recording joins of tree nodes into rel_dict
        inputs: node:              the current visiting tree node 
             rel_dict:           dict form of the upper tri join matrix
             alias2table:         the corresponding alias-relation pairs' dictionary that we can directly get from dataset
             sub_query_rels:       list with the relation / alias pairs that can't direct find in rel_names, with count, eg: ['n1', 'title', 1]
        '''

        assert is_join(node)
        cond = node.join
        if cond != 'NA':
            # rel_list record the relations involved in the join conditions
            rel_list = []
            count = 0
            condleft = list(filter(None, re.split(r'[(= )]', cond)))[0]
            condright = list(filter(None, re.split(r'[(= )]', cond)))[1]
            count = list(filter(None, re.split(r'[(= )]', cond))).count(condleft)
            
            # we use alias2table to find relation by its alias, if alias exists    
            if alias2table:
                if condleft.split('.')[0] in self.REL_NAMES:
                    rel_list.append(condleft.split('.')[0])
                elif condleft.split('.')[0] in alias2table.keys() and alias2table[condleft.split('.')[0]] != None:
                    rel_list.append(alias2table[condleft.split('.')[0]])
                if condright.split('.')[0] in self.REL_NAMES:
                    rel_list.append(condright.split('.')[0])
                elif condright.split('.')[0] in alias2table.keys() and alias2table[condright.split('.')[0]] != None:
                    rel_list.append(alias2table[condright.split('.')[0]])
                    
            else:
                if condleft.split('.')[0] in self.REL_NAMES:
                    rel_list.append(condleft.split('.')[0])
                elif condright.split('.')[0] in self.REL_NAMES:
                    rel_list.append(condright.split('.')[0])
                    
        #     tpcds got alias = self, so cond may have tableA = alias(tableA) situation, we ignore it by len(set(rel_list)) != 1
            if len(rel_list) == 2 and len(set(rel_list)) != 1:
            #    situation when both relations can be successfully parsed
                try:
                    rel_dict[rel_list[0]][rel_list[1]] += count
                except:
                    rel_dict[rel_list[1]][rel_list[0]] += count
                    
        #    if len(rel_list) != 2 (= 1 or = 0), it comes to the condition that exists subqueries and alias not esist or can not match
            elif len(rel_list) != 2:
                cond_split = list(filter(None, re.split(r'[( = ,)]', cond)))
                a = []
                for s in cond_split:
                    if '.' in s:
                        a.append(s)
                # get the alias names
                alias1 = list(filter(None, re.split(r'[.]', a[0])))[0]
                attr1 = list(filter(None, re.split(r'[.]', a[0])))[1]
                alias2 = list(filter(None, re.split(r'[.]', a[1])))[0]
                attr2 = list(filter(None, re.split(r'[.]', a[1])))[1]

                count = re.split(r'[(=. )]',cond).count(alias1)

                # subrels is list of all tables involved in this join, found by indexing the original tables with the cond attributes
                subrels1 = [alias1]
                subrels2 = [alias2]
                for subrel in self.REL_NAMES:
                    if attr1 in self.REL_ATTR_LIST_DICT[subrel]:
                        if subrel != alias1:
                            subrels1.append(subrel)
                    if attr2 in self.REL_ATTR_LIST_DICT[subrel]:
                        if subrel != alias2:
                            subrels2.append(subrel)  

                # we ignore some situations the relations can not be parsed eg: tpcds qid=44 we have filter 'alias+rank=alias+rank'
                # then recird the rest situations down and keep these join records to add on to upper tri later in func query_encode
                if not ((len(subrels1) == 1 and subrels1[0] not in self.REL_NAMES) or (len(subrels2) == 1 and subrels2[0] not in self.REL_NAMES)):
                    sub_query_rels.append(subrels1)
                    sub_query_rels.append(subrels2)
                    sub_query_rels.append(count)

#         print(sub_query_rels)






    def __featurize_pred_one_hot(self, node, rel_attr_dict):
        '''
        encode the predicates of the query by one-hot
        inputs: node:              the current visiting tree node 
             rel_attr_dict:        the relation-attribute dict of the dataset
        '''
        assert is_scan(node)
        rel = node.table
        fils = node.filters

#        index the location of the predicate column in the dict, replace it with 1 if it is not already 1 (used by other scans)
        if rel == None:
            rels = []
            for f in fils:
                for r in self.REL_NAMES:
                    if f[0] == self.REL_ATTR_LIST_DICT[r]:
                        rels.append(r)
            rels = list(set(rels)) 
            for relation in rels:
                for attr in rel_attr_dict[relation]:
                    if isinstance(attr, str):
                        for fil in fils:
                            if fil[0] == attr:
                                rel_attr_dict[relation][rel_attr_dict[relation].index(attr)] = (node.card_est - self.rows_min) / (self.rows_max - self.rows_min)
                                
        #  subquery scan has a 'Filter' type but has no relation name, has to be indexed from the filter
        else:
            for attr in rel_attr_dict[rel]:
                if isinstance(attr, str):
                    for fil in fils:
                        #  special case for 'Index Name' being 'title_pkey'
                        if 'pkey' in fil[0]:
                            fil = fil.split('.')[0] + '.id'
                        if fil[0] == attr:
                            if attr in rel_attr_dict[rel]:
                                rel_attr_dict[rel][rel_attr_dict[rel].index(attr)] = (node.card_est - self.rows_min) / (self.rows_max - self.rows_min)
                        # special case for reading the stats dataset attribute, being a different form from tpch, tpcds and imdb
                        elif fil[0].split('.')[-1] == attr:
                            if attr in rel_attr_dict[rel]:
                                rel_attr_dict[rel][rel_attr_dict[rel].index(attr)] = (node.card_est - self.rows_min) / (self.rows_max - self.rows_min)




    def plan_to_feature_query(self, root, rel_dict, rel_attr_dict, sub_query_rels, alias2table):
        '''
        transversing the plan tree and update the encoding of each node from bottom to top
        inputs: root:              the root node of plan tree 
             rel_dict:           dict form of the upper tri join matrix
             rel_attr_dict:        the relation-attribute dict of the dataset
             sub_query_rels:       list with the relation / alias pairs that can't direct find in rel_names, with count, eg: ['n1', 'title', 1]
             alias2table:         the corresponding alias-relation pairs' dictionary that we can directly get from dataset
        '''        

            
        if is_join(root):
            assert len(root.children) >= 2
            self.__featurize_qjoin(root, rel_dict, sub_query_rels, alias2table)
            left = self.plan_to_feature_query(root.children[0], rel_dict, rel_attr_dict, sub_query_rels, alias2table)
            right = self.plan_to_feature_query(root.children[1], rel_dict, rel_attr_dict, sub_query_rels, alias2table)
            # tpch qid=15 got 3 children for the first merge join :)
            if len(root.children) == 3:
                mid = self.plan_to_feature_query(root.children[2], rel_dict, rel_attr_dict, sub_query_rels, alias2table)

        if is_scan(root):
            self.__featurize_pred_one_hot(root, rel_attr_dict)

        if len(root.children) == 1:
        # ignore and continue at other not join/scan tree nodes. eg: aggragate, sort, etc
            return self.plan_to_feature_query(root.children[0], rel_dict, rel_attr_dict, sub_query_rels, alias2table)


    def query_encode(self, root):
        '''
        encode the plan tree in the query level
        inputs: root:              the root node of plan tree 
        '''     

        # initialize the dicts needed
        rel_attr_dict = copy.deepcopy(self.REL_ATTR_LIST_DICT)
        alias2table = self.alias2table
        sub_query_rels = []
        
        rel_dict = {}
        for i in range(len(self.REL_NAMES)-1):
            rel_dict[self.REL_NAMES[i]] = {}
            for j in range(i+1, len(self.REL_NAMES)):
                rel_dict[self.REL_NAMES[i]][self.REL_NAMES[j]] = 0
        
        # encode at query level by transvering the plan tree, sub_query_rels, rel_dict and rel_attr_dict are auto-updated
        self.plan_to_feature_query(root, rel_dict, rel_attr_dict, sub_query_rels, alias2table)

#         print('reldict: ', rel_dict)

        # start of special situation
        # this part solves the situation for join encoding with rel_list < 2, which means we have dismatch relations that can not be found
        # the situation will be stored in list 'sub_query_rels'
        for p in range(int(len(sub_query_rels)/3)):
            subrels1 = sub_query_rels[3*p]
            subrels2 = sub_query_rels[3*p+1]
            count = sub_query_rels[3*p+2]
            if len(subrels1) != 1:
                subrels1.pop(0)
                r = subrels1[-1]
                for potential_related in self.REL_NAMES:
                    try:
                        if rel_dict[r][potential_related] != 0:
                            subrels1.append(potential_related)
                    except:
                        try:
                            if rel_dict[potential_related][r] != 0:
                                subrels1.append(potential_related)
                        except:
                            # if this relation name is not the match, simply continue to try to match with next relation name
                            continue
            if len(subrels2) != 1:
                subrels2.pop(0)
                r = subrels2[-1]
                for potential_related in self.REL_NAMES:
                    try:
                        if rel_dict[r][potential_related] != 0:
                            subrels2.append(potential_related)
                    except:
                        try:
                            if rel_dict[potential_related][r] != 0:
                                subrels2.append(potential_related)
                        except:
                            continue
            for m in subrels1:
                for n in subrels2:
                    if m != n:
                        try:
                            rel_dict[m][n] += count
                        except:
                            rel_dict[n][m] += count
        # end of special situation                    
                            
        # 'query_part1' is arr form of upper tri for the relation join matrix, and 'query_part2' is arr for used predicates counts
        query_part1 = []
        query_part2 = []

#         print('rel_dict: ', rel_dict)
#         print('rel_attr_dict: ', rel_attr_dict)

        for rel in self.REL_NAMES[:-1]:
            for k in rel_dict[rel].keys():
                query_part1.append(rel_dict[rel][k])

        for rel in self.REL_NAMES:
            for k in rel_attr_dict[rel]:
                if isinstance(k, int):
                    query_part2.append(k)
                else:
                    query_part2.append(0)
                    
        return query_part1 + query_part2


#     function moved to TreeFeaturizer
#     def query_flat(self, root):
#         '''
#         flatten the encoded query level vector to combine with the plan level vector later
#         inputs: root:              the root node of plan tree 
#         '''  
#         out_vec = self.query_encode(root)
#         out_vec = FloatTensor(out_vec)
#         q_conv = nn.Sequential(nn.Linear(len(out_vec), 64),
#                                nn.Linear(64, 128),
#                                nn.Linear(128, 64),
#                                nn.Linear(64, 32))

#         return q_conv(out_vec)



    ################################################ plan level encoding ################################################


    def __featurize_join_plan(self, node):
        '''
        encode the plan tree joins
        inputs: node:              the current visiting tree node 
        '''     
        assert is_join(node)
        arr = np.zeros(len(self.ALL_TYPES))
        if node.nodeType in self.ALL_TYPES:
            arr[self.ALL_TYPES.index(node.nodeType)] = 1
        else:
            arr[self.ALL_TYPES.index("Other Join")] = 1
        return arr

    def __featurize_scan_plan(self, node):
        '''
        encode the plan tree scans
        inputs: node:              the current visiting tree node 
        '''     
        assert is_scan(node)
        arr = np.zeros(len(self.ALL_TYPES))
        if node.nodeType in self.ALL_TYPES and node.table:
            arr[len(JOIN_TYPES) + len(SCAN_TYPES) * self.REL_NAMES.index(node.table) + SCAN_TYPES.index(node.nodeType)] = 1  
    #     CTE scan we assume that all entries are 0
        return arr



    def plan_to_feature_tree(self, root):
        '''
        transversing the plan tree and update the encoding of each node from bottom to top
        inputs: root:              the root node of plan tree 
        '''   
        
        if is_join(root):
            assert len(root.children) >= 2
            my_vecj = self.__featurize_join_plan(root)

            left = self.plan_to_feature_tree(root.children[0])
            right = self.plan_to_feature_tree(root.children[1])

            if isinstance(left, tuple):
                leftchild = copy.deepcopy(left[0])
            else:
                leftchild = copy.deepcopy(left)
            if isinstance(right, tuple):
                rightchild = copy.deepcopy(right[0])
            else:
                rightchild = copy.deepcopy(right)
            leftchild[0] = leftchild[1] = rightchild[0] = rightchild[1] = 0
            my_vecj = my_vecj + leftchild + rightchild

            for i in range(len(JOIN_TYPES)):
                if my_vecj[i] > 1:
                    my_vecj[i] = 1

    #         when a join has more than 2 children, consider as a series of multiple joins with the same type
            if len(root.children) == 3:
                mid = self.plan_to_feature_tree(root.children[2])
                if isinstance(mid, tuple):
                    midchild = copy.deepcopy(mid[0])
                else:
                    midchild = copy.deepcopy(mid)
                midchild[0] = midchild[1] = 0
                my_vecj_root = my_vecj + midchild
                return (my_vecj_root, mid, (my_vecj, left, right))

            return (my_vecj, left, right)

        if is_scan(root):
            if root.children == []:
                return self.__featurize_scan_plan(root)
            else:
    #           select only the first node, which is the root of the subquery
                if isinstance(self.plan_to_feature_tree(root.children[0]), tuple):
    #               when subquery is a complex query
                    my_vecs = self.plan_to_feature_tree(root.children[0])[0]
                else:
    #               when subquery has only one scan
                    my_vecs = self.plan_to_feature_tree(root.children[0])
                for i in range(len(self.REL_NAMES)):
                    if 1 in my_vecs[(len(JOIN_TYPES) + len(SCAN_TYPES) * i) : (len(JOIN_TYPES) + len(SCAN_TYPES) * (i + 1) - 1)]:
                        my_vecs[len(JOIN_TYPES) + len(SCAN_TYPES) * (i + 1) - 1] = 1
    #           tpch has subqueries that are not considered into the subquery scan type, consists of 2 seq scans
                if root.table:
                    ind = self.REL_NAMES.index(root.table)
                    my_vecs[len(JOIN_TYPES) + len(SCAN_TYPES) * ind] = 1
                return my_vecs

    #           include all the subquery
    #             my_vec = plan_to_feature_tree(children[0])
    #             for i in range(len(self.__relations)):
    #                 if 1 in my_vec[0][(len(JOIN_TYPES) + len(LEAF_TYPES) * i) : (len(JOIN_TYPES) + len(LEAF_TYPES) * (i + 1) - 1)]:
    #                     my_vec[0][len(JOIN_TYPES) + len(LEAF_TYPES) * (i + 1) - 1] = 1
    #             return my_vec


        if len(root.children) == 1:
            return self.plan_to_feature_tree(root.children[0])




class TreeFeaturizer:
    def __init__(self, rel_names, rel_attr_list_dict, alias2table):
        self.__tree_builder = None 
        self._relations = rel_names
        self._attr_dict = rel_attr_list_dict    
        self.NUM_REL = len(self._relations)
        self.ALL_TYPES = JOIN_TYPES + self.NUM_REL * SCAN_TYPES
        ##### add qflat
        
        self.alias2table = alias2table
        
    def fit(self, trees):
        rows_min, rows_max = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(self._relations, self._attr_dict, self.alias2table, rows_min, rows_max)

    def transform(self, trees):
        plan_trees = [self.__tree_builder.plan_to_feature_tree(x) for x in trees]
        q_vecs = [self.__tree_builder.query_encode(x) for x in trees]
        
        return plan_trees, q_vecs

    def num_operators(self):
        return len(self.ALL_TYPES)


