import numpy as np

JOIN_TYPES = ['Hash Join', 'Nested Loop', 'Merge Join', 'Other Join']
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan", 
              'Bitmap Heap Scan', 'Subquery Scan']
ALL_TYPES = JOIN_TYPES + SCAN_TYPES


class Batch():
    def __init__(self,trees, idxes):
        self.trees = trees
        self.idxes = idxes
    def to(self,dev):
        self.trees = self.trees.to(dev)
        self.idxes = self.idxes.to(dev)
        return self


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
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.FloatTensor(targets).reshape(-1, 1)
    flat_trees, indexes = prepare_trees(trees, features, left_child, right_child)
    return Batch(flat_trees, indexes), targets

from TreeConvolution.util import *
def prepare_trees(trees, transformer, left_child, right_child, cuda=False):
    flat_trees = [flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # flat trees is now batch x max tree nodes x channels
    flat_trees = flat_trees.transpose(1, 2)

    indexes = [tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    return (flat_trees, indexes)



class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return len(node.children) >= 2

def is_scan(node):
    if node.children == []:
        return True
    return node.nodeType in SCAN_TYPES


class TreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)

#     def __relation_name(self, node):
#         if "Relation Name" in node:
#             return node["Relation Name"]

#         if node["Node Type"] == "Bitmap Index Scan":
#             # find the first (longest) relation name that appears in the index name
#             name_key = "Index Name" if "Index Name" in node else "Relation Name"
#             if name_key not in node:
#                 print(node)
#                 raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
#             for rel in self.__relations:
#                 if rel in node[name_key]:
#                     return rel

#             raise TreeBuilderError("Could not find relation name for bitmap index scan")

#         raise TreeBuilderError("Cannot extract relation type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        arr = np.zeros(len(ALL_TYPES))
        if node.nodeType in ALL_TYPES:
            arr[ALL_TYPES.index(node.nodeType)] = 1
        else:
            arr[ALL_TYPES.index("Other Join")] = 1
        return np.concatenate((arr, self.__stats(node)))

    def __featurize_scan(self, node):
        assert is_scan(node)
        arr = np.zeros(len(ALL_TYPES))
        if node.nodeType in ALL_TYPES:
            arr[ALL_TYPES.index(node.nodeType)] = 1
        return (np.concatenate((arr, self.__stats(node))),
                node.table)

    def plan_to_feature_tree(self, root):
        children = root.children

       
        if is_join(root):
            assert len(children) >= 2
            my_vec = self.__featurize_join(root)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            
            if len(children) == 3:
                mid = self.plan_to_feature_tree(children[2])
    #             print(mid)
                return (my_vec, mid, (my_vec, left, right))
    
            return (my_vec, left, right)

        if is_scan(root):
            if root.children == []:
                return self.__featurize_scan(root)
            else:
    #           select only the first node, which is the root of the subquery
                if isinstance(self.plan_to_feature_tree(root.children[0]), tuple):
    #               when subquery is a complex query
                    my_vec = self.plan_to_feature_tree(root.children[0])[0]
                else:
    #               when subquery has only one scan
                    my_vec = self.plan_to_feature_tree(root.children[0])
                my_vec[-1] = 1
                return my_vec

        
        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])


        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))

def norm(x, lo, hi):
    return (np.log(x + 1) - lo) / (hi - lo)

def get_buffer_count_for_leaf(leaf, buffers):
    total = 0
    if leaf.table:
        total += buffers.get(leaf.table, 0)

    if leaf.index:
        total += buffers.get(leaf.index, 0)

    return total

class StatExtractor:
    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        res = []
        for f, lo, hi in zip([inp.buffers, inp.cost_est, inp.card_est], self.__mins, self.__maxs):
            if not f:
                res.append(0)
            else:
                res.append(norm(f, lo, hi))
        return res

def get_plan_stats(roots):
    costs = []
    rows = []
    bufs = []
    
    def recurse(n):
        costs.append(n.cost_est)
        rows.append(n.card_est)
        if n.buffers:
            bufs.append(n.buffers)

        if n.children != []:
            for child in n.children:
                recurse(child)

    for root in roots:
        recurse(root)

    costs = np.array(costs)
    rows = np.array(rows)
    bufs = np.array(bufs)
    
    costs = np.log(costs + 1)
    rows = np.log(rows + 1)
    bufs = np.log(bufs + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    bufs_min = np.min(bufs) if len(bufs) != 0 else 0
    bufs_max = np.max(bufs) if len(bufs) != 0 else 0

#     if len(bufs) != 0:
    return StatExtractor(
        ["Buffers", "Total Cost", "Plan Rows"],
        [bufs_min, costs_min, rows_min],
        [bufs_max, costs_max, rows_max]
    )
#     else:
#         return StatExtractor(
#             ["Total Cost", "Plan Rows"],
#             [costs_min, rows_min],
#             [costs_max, rows_max]
#         )
        

def get_all_relations(data):
    all_rels = []
    
    def recurse(root):
        if root.table:
            yield root.table

        if root.children != []:
            for child in root.children:
                yield from recurse(child)

    for root in data:
        all_rels.extend(list(recurse(root)))
        
    return set(all_rels)

def get_featurized_trees(data):
    all_rels = get_all_relations(data)
    stats_extractor = get_plan_stats(data)

    t = TreeBuilder(stats_extractor, all_rels)
    trees = []

    for root in data:
        tree = t.plan_to_feature_tree(root)
        trees.append(tree)
            
    return trees

def _attach_buf_data(root):
    if not root.buffers:
        return

    buffers = root.buffers

    def recurse(n):
        if n.children != []:
            for child in n.children:
                recurse(child)
            return
        
        # it is a leaf
        n.buffers = get_buffer_count_for_leaf(n, buffers)

    recurse(root)

class TreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        for t in trees:
            _attach_buf_data(t)
        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):
        for t in trees:
            _attach_buf_data(t)
        return [self.__tree_builder.plan_to_feature_tree(x) for x in trees]

    def num_operators(self):
        return len(ALL_TYPES)
