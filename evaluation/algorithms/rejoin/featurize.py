import numpy as np
import torch
from torch import tensor
import copy
import re


# edited by lz
JOIN_TYPES = ['Hash Join', 'Nested Loop', 'Merge Join', 'Other Join']
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan",
              'Bitmap Heap Scan', 'Subquery Scan']


def is_join(node):
    return len(node.children) >= 2


def is_scan(node):
    if node.children == []:
        return True
    return node.nodeType in SCAN_TYPES


def collate(x):
    in_vecs = []
    targets = []

    for statevec, target in x:
        tree_vec = statevec.tree_structure
        join_vec = []
        for i in statevec.join_predicates:
            join_vec += i
        selec_vec = statevec.selection_predicates

        # in_vec = torch.FloatTensor(tree_vec + join_vec + selec_vec)
        in_vec = tree_vec + join_vec + selec_vec
#         print(in_vec)
        in_vecs.append(in_vec)
        targets.append(target)

    in_vecs = torch.FloatTensor(in_vecs)
    targets = torch.FloatTensor(targets).reshape(-1,1)

#     return Batch(in_vecs), targets
    return in_vecs, targets





class StateVector:
    def __init__(self, root, REL_NAMES, REL_ATTR_LIST_DICT, alias2table):

        self.root = root
        self.REL_NAMES = REL_NAMES
        self.REL_ATTR_LIST_DICT = REL_ATTR_LIST_DICT
        self.alias2table = alias2table
        self.aliases = {}

        self.tree_structure, self.join_predicates, self.selection_predicates = self.query_encode(self.root)


    # def __featurize_qjoin(self, node, rel_dict, sub_query_rels, height_vec, height):
    #
    #     assert is_join(node)
    #
    #     cond = node.join
    #     if cond != 'NA':
    #         # print(cond)
    #         rel_list = []
    #         count = 0
    #         condleft = list(filter(None, re.split(r'[(= )]', cond)))[0]
    #         condright = list(filter(None, re.split(r'[(= )]', cond)))[1]
    #         count = list(filter(None, re.split(r'[(= )]', cond))).count(condleft)
    # #         print(condleft, condright)
    #         for rel in self.REL_NAMES:
    # #             print(self.REL_ATTR_LIST_DICT[rel])
    #             if rel in re.split(r'[(=. )]',cond):
    # #                 count = re.split(r'[(=. )]',cond).count(rel)
    #                 rel_list.append(rel)
    #             elif condleft in self.REL_ATTR_LIST_DICT[rel]:
    #                 rel_list.append(rel)
    #             elif condright in self.REL_ATTR_LIST_DICT[rel]:
    #                 rel_list.append(rel)
    # #         print(rel_dict)
    #         if len(rel_list) == 2:
    #             rel_dict[rel_list[0]][rel_list[1]] += count
    #             rel_dict[rel_list[1]][rel_list[0]] += count
    #             height_vec[self.REL_NAMES.index(rel_list[0])] = 1 / (2 ** height)
    #             height_vec[self.REL_NAMES.index(rel_list[1])] = 1 / (2 ** height)
    # #       if len(rel_list) != 2 (= 1 or = 0), it comes to the condition that exists subqueries
    # #       that alias not in REL_NAMESss
    #         else:
    # #             print(cond)
    #             cond_split = list(filter(None, re.split(r'[( = ,)]', cond)))
    #             a = []
    #             for s in cond_split:
    #                 if '.' in s:
    #                     a.append(s)
    # #             print(a)
    #             alias1 = list(filter(None, re.split(r'[.]', a[0])))[0]
    #             attr1 = list(filter(None, re.split(r'[.]', a[0])))[1]
    #             alias2 = list(filter(None, re.split(r'[.]', a[1])))[0]
    #             attr2 = list(filter(None, re.split(r'[.]', a[1])))[1]
    #
    # #             print(attr1)
    # #             print(attr2)
    #             count = re.split(r'[(=. )]',cond).count(alias1)
    #
    #             subrels1 = [alias1]
    #             subrels2 = [alias2]
    #             for subrel in self.REL_NAMES:
    # #                 print(self.REL_ATTR_LIST_DICT[subrel][2])
    # #                 print(attr1 in self.REL_ATTR_LIST_DICT[subrel])
    #                 if attr1 in self.REL_ATTR_LIST_DICT[subrel]:
    #                     if subrel != alias1:
    #                         subrels1.append(subrel)
    #                 if attr2 in self.REL_ATTR_LIST_DICT[subrel]:
    #                     if subrel != alias2:
    #                         subrels2.append(subrel)
    #
    # #             print(subrels1)
    # #             print(subrels2)
    #
    # #             some situations the relations can not be parsed eg: tpcds qid=44 we have filter 'alias+rank=alias+rank'
    #             if not ((len(subrels1) == 1 and subrels1[0] not in self.REL_NAMES) or (len(subrels2) == 1 and subrels2[0] not in self.REL_NAMES)):
    #
    #                 sub_query_rels.append(subrels1)
    #                 sub_query_rels.append(subrels2)
    #                 sub_query_rels.append(count)
    #                 sub_query_rels.append(height)


    def __featurize_qjoin(self, node, rel_dict, height_vec, height, alias2table):
        '''
        encode the upper tri of join matrix by recording joins of tree nodes into rel_dict
        inputs: node:              the current visiting tree node
             rel_dict:           dict form of the upper tri join matrix
             alias2table:         the corresponding alias-relation pairs' dictionary that we can directly get from dataset
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

                # print(rel_list[0] in self.REL_NAMES)
                # print(height_vec)
                # print(1 / (2 ** sum(height)))
                # print(height_vec[self.REL_NAMES.index(rel_list[0])])
                height_vec[self.REL_NAMES.index(rel_list[0])] = 1 / (2 ** height)
                height_vec[self.REL_NAMES.index(rel_list[1])] = 1 / (2 ** height)





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
                                rel_attr_dict[relation][rel_attr_dict[relation].index(attr)] = 1

        #  subquery scan has a 'Filter' type but has no relation name, has to be indexed from the filter
        else:
            for attr in rel_attr_dict[rel]:
                if isinstance(attr, str):
                    for fil in fils:
                        # print(fil[0])
                        # print(fil[0].split('.')[-1])
                        # print(attr)
                        # print('-------------------------------')
                        #  special case for 'Index Name' being 'title_pkey'
                        if 'pkey' in fil[0]:
                            fil = fil.split('.')[0] + '.id'
                        if fil[0] == attr:
                            if attr in rel_attr_dict[rel]:
                                rel_attr_dict[rel][rel_attr_dict[rel].index(attr)] = 1
                        # special case for reading the stats dataset attribute, being a different form from tpch, tpcds and imdb
                        elif fil[0].split('.')[-1] == attr:
                            if attr in rel_attr_dict[rel]:
                                rel_attr_dict[rel][rel_attr_dict[rel].index(attr)] = 1
                        elif fil[0].split('.')[-1] == attr.lower():
                            if attr in rel_attr_dict[rel]:
                                rel_attr_dict[rel][rel_attr_dict[rel].index(attr)] = 1







    def plan_to_feature_query(self, root, rel_dict, rel_attr_dict, height_vec, height, alias2table):


        if is_join(root):
            assert len(root.children) >= 2
            height += 1
#             print(height)
            self.__featurize_qjoin(root, rel_dict, height_vec, height, alias2table)
            left = self.plan_to_feature_query(root.children[0], rel_dict, rel_attr_dict, height_vec, height, alias2table)
            right = self.plan_to_feature_query(root.children[1], rel_dict, rel_attr_dict, height_vec, height, alias2table)
    #         tpch qid=15 got 3 children for the first merge join :)
            if len(root.children) == 3:
                mid = self.plan_to_feature_query(root.children[2], rel_dict, rel_attr_dict, height_vec, height, alias2table)


        if is_scan(root):
    #         assert not children
            self.__featurize_pred_one_hot(root, rel_attr_dict)


        if len(root.children) == 1:
            return self.plan_to_feature_query(root.children[0], rel_dict, rel_attr_dict, height_vec, height, alias2table)

    #     raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


    def query_encode(self, root):

    #     rel_attr_d'Index Name': 'title_pkey'ict = dict(self.REL_ATTR_LIST_DICT)
        rel_attr_dict = copy.deepcopy(self.REL_ATTR_LIST_DICT)
        alias2table = self.alias2table

        rel_dict = {}

        height = 0
        height_vec = [0] * len(self.REL_NAMES)


        for i in range(len(self.REL_NAMES)):
            rel_dict[self.REL_NAMES[i]] = {}
            for rname in self.REL_NAMES:
                rel_dict[self.REL_NAMES[i]][rname] = 0

        self.plan_to_feature_query(root, rel_dict, rel_attr_dict, height_vec, height, alias2table)

    #     print(sub_query_rels)

    #     for p in range(int(len(sub_query_rels)/4)):
    # #       add on count to all related relations
    #         subrels1 = sub_query_rels[4*p]
    #         subrels2 = sub_query_rels[4*p+1]
    #         count = sub_query_rels[4*p+2]
    #         height = sub_query_rels[4*p+3]
    #
    #         if len(subrels1) != 1:
    #             subrels1.pop(0)
    #             r = subrels1[-1]
    #             for potential_related in self.REL_NAMES:
    #                 if rel_dict[r][potential_related] != 0:
    #                     subrels1.append(potential_related)
    #                 elif rel_dict[potential_related][r] != 0:
    #                     subrels1.append(potential_related)
    #
    #
    #         if len(subrels2) != 1:
    #             subrels2.pop(0)
    #             r = subrels2[-1]
    #             for potential_related in self.REL_NAMES:
    #                 if rel_dict[r][potential_related] != 0:
    #                     subrels2.append(potential_related)
    #                 elif rel_dict[potential_related][r] != 0:
    #                     subrels2.append(potential_related)
    #
    #         for m in subrels1:
    #             for n in subrels2:
    #                 if m != n:
    #                     rel_dict[m][n] += count
    #                     rel_dict[n][m] += count
    #                     height_vec[self.REL_NAMES.index(m)] = 1 / (2 ** height)
    #                     height_vec[self.REL_NAMES.index(n)] = 1 / (2 ** height)

        tree_struct_vec = height_vec
        join_pred_vec = [[0 for x in range(len(self.REL_NAMES))] for y in range(len(self.REL_NAMES))]

        select_pred_vec = []

        for k1 in rel_dict.keys():
            for k2 in rel_dict[k1].keys():
                if rel_dict[k1][k2] != 0:
                    join_pred_vec[self.REL_NAMES.index(k1)][self.REL_NAMES.index(k2)] = 1
                else:
                    join_pred_vec[self.REL_NAMES.index(k1)][self.REL_NAMES.index(k2)] = rel_dict[k1][k2]

        for rel in self.REL_NAMES:
            for k in rel_attr_dict[rel]:
                if isinstance(k, int):
                    select_pred_vec.append(1)
                else:
                    select_pred_vec.append(0)

        # print(select_pred_vec)
        return tree_struct_vec, join_pred_vec, select_pred_vec













    def vectorize(self):
        states = dict()
        states["tree_structure"] = np.array(self.tree_structure, dtype=float).flatten()
        states["join_predicates"] = np.array(self.join_predicates).flatten()
        states["selection_predicates"] = np.array(self.selection_predicates)

        return states

    def print_state(self):  # Print for debugging
        print("\nTree-Structure:\n")
        print(np.array(self.tree_structure))
        print(np.array(self.tree_structure).shape)
        print("\nJoin-Predicates:\n")
        print(np.array(self.join_predicates))
        print(np.array(self.join_predicates).shape)
        print("\nSelection-Predicates:\n")
        print(np.array(self.selection_predicates))
        print(np.array(self.selection_predicates).shape)

#     def print_joined_attrs(self):
#         self.pp.pprint(self.joined_attrs)

#     def print_query(self):
#         self.pp.pprint(self.query_ast)

#     def print_aliases(self):
#         self.pp.pprint(self.aliases)

#     def print_alias_to_relations(self):
#         self.pp.pprint(self.alias_to_relations)
