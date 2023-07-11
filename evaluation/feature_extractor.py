import pandas as pd
import numpy as np
from utils import Normalizer
from pyparsing import nestedExpr
from datetime import datetime
import time
import json

class TreeNode:

    def __init__(self, dictionary):
        self.level = 0 # root

        self.cost = None
        self.card = None

        self.cost_est = None
        self.card_est = None

        self.table = None
        self.alias = None
        self.index = None

        self.filters = None
        self.join = None

        self.nodeType = None
        self.nodeParallel = None ## for AIMeetsAI
        self.width = None

        self.children = []
        self.parent = None

        # added for bao
        self.buffers = None   
        
        # for plan cost
        self.hash = None
        self.parent_rel = None
        self.join_type = None
        self.sort_key = None
        self.sort_method = None
        self.strategy = None
        self.para_aware = None
        self.command = None
        self.subplan_name = None

        self.__dict__.update(dictionary)

    def update(self,dictionary):
        self.__dict__.update(dictionary)
        
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filters, self.join, len(self.children))

    def __repr__(self):
        return self.__str__()

    def print_nested(self): 
        print('--'*self.level+ self.__str__())
        for k in self.children: 
            k.print_nested()


class DatasetInfo:
    
    def __init__(self, dictionary):
        self.max_card = None
        self.min_card = 1.001
        self.max_cost = None
        self.min_cost = 0.001 # ms
        self.max_cost_est = None
        self.min_cost_est = 0.01
        
        
        self.card_norm = None
        self.cost_norm = None
        self.cost_est_norm = None
        # for plan cost
        self.width_norm = None
        
        self.nodeTypes = []
        self.nodeParallels = []
        
        self.tables = []
        self.aliass = []
        self.indexes = []
        self.joins = []
        
        self.alias2table = {}
        self.table2alias = {}

        self.max_filters = 0
        
        self.column_min_max_vals = {}
        self.columns = []
        
        self.__dict__.update(dictionary)
    
    def update(self, dictionary):
        self.__dict__.update(dictionary)
    
    def get_columns(self, column_min_max_vals):
        self.column_min_max_vals = column_min_max_vals
        self.columns = list(column_min_max_vals.keys())
    
    def construct_from_plans(self, plans):
        # for plan cost
        max_card = max_cost = max_cost_est = max_width = 0
        min_card = min_cost = min_cost_est = min_width = 100
        max_filters = 0

        for root in plans:

            toVisit = [root] #bfs
            while toVisit: 
                node = toVisit.pop(0)
                for child in node.children:
                    toVisit.append(child)

                if node.filters is not None:
                    max_filters = max(max_filters, len(node.filters))

                max_card = max(max_card, node.card, node.card_est)
                max_cost = max(max_cost, node.cost)
                max_cost_est = max(max_cost_est, node.cost_est)

                min_card = min(min_card, node.card, node.card_est)
                min_cost = min(min_cost, node.cost)
                min_cost_est = min(min_cost_est, node.cost_est)

                if node.nodeType not in self.nodeTypes:
                    self.nodeTypes.append(node.nodeType)
                if node.nodeParallel not in self.nodeParallels:
                    self.nodeParallels.append(node.nodeParallel)
                
                
                if node.index is not None and node.index not in self.indexes:
                    self.indexes.append(node.index)
                if node.join is not None and node.join not in self.joins:
                    self.joins.append(node.join)
                if node.table is not None and node.table not in self.tables:
                    self.tables.append(node.table)
                if node.alias is not None and node.alias not in self.aliass: # may have problem for subquery
                    self.aliass.append(node.alias)
                    self.alias2table[node.alias] = node.table
                    self.table2alias[node.table] = node.alias
                    
        self.max_card, self.max_cost, self.max_cost_est = max_card, max_cost, max_cost_est
        self.min_card, self.min_cost, self.min_cost_est = min_card, min_cost, min_cost_est
        
        self.card_norm = Normalizer(np.log(float(min_card) + 0.001), np.log(float(max_card) + 0.001))
        self.cost_norm = Normalizer(np.log(float(min_cost) + 0.001), np.log(float(max_cost) + 0.001))
        self.cost_est_norm = Normalizer(np.log(float(min_cost_est) + 0.001), np.log(float(max_cost_est) + 0.001))
        # for plan cost
        self.width_norm = Normalizer(np.log(float(min_width) + 0.001), np.log(float(max_width) + 0.001))

        self.max_filters = max_filters

def getAlias(node):
    alias = None
    if 'Alias' in node:
        alias = node['Alias']
    else:
        n = node
        while 'parent' in n:
            n = n['parent']
            if 'Alias' in n:
                alias = n['Alias']
                break
    return alias

def extractNode(node):    
    d = {
        'nodeType' : node['Node Type'],
        'card'     : node['Actual Rows'],
        'card_est' : node['Plan Rows'],
        'nodeParallel' : node['Node Type'] + '_' + str(node['Parallel Aware']),
        'width'    : node['Plan Width'],
    }
    
    alias = getAlias(node)
    d['alias'] = alias
    
    conds = get_conditions(node, alias)
    join, filters = conds['join'], conds['filters']

    d['filters'] = filters
    d['join'] = join

    d['cost'] = node['Actual Total Time']
    d['cost_est'] = node['Total Cost']
    
    if 'Index Name' in node:
        d['index'] = node['Index Name']
    
    if 'Relation Name' in node:
        d['table'] = node['Relation Name']

#     adding for bao
    if 'Buffers' in node:
        d['buffers'] = node['Buffers']
        
    # for plan cost
    if 'Hash Buckets' in node:
        d['hash'] = node['Hash Buckets']
        
    if 'Parent Relationship' in node:
        d['parent_rel'] = node['Parent Relationship'].lower()
        
    if 'Join Type' in node:
        d['join_type'] = node['Join Type'].lower()
        
    if 'Sort Key' in node:
        d['sort_key'] = node['Sort Key']
        
    if 'Sort Method' in node:
        d['sort_method'] = node['Sort Method'].lower()
        
    if 'Strategy' in node:
        d['strategy'] = node['Strategy'].lower()
        
    if 'Parallel Aware' in node:
        d['para_aware'] = node['Parallel Aware']
        
    if 'Command' in node:
        d['command'] = node['Command']
        
    if 'Subplan Name' in node:
        d['subplan_name'] = node['Subplan Name']
         
    return d

def traversePlan(root, level=0):
    if 'Plan' in root:
        root = root['Plan']
    root_node = TreeNode(extractNode(root))
    root_node.level = level
    if 'Plans' in root:
        for child in root['Plans']:
            child['parent'] = root
            node = traversePlan(child, level+1)
            node.parent = root_node
            root_node.children.append(node)
    return root_node   

def get_conditions(json_node, table):
    conds = set()
    ####################################################################
    # join conds
    if 'Hash Cond' in json_node:
        conds.add(json_node['Hash Cond'])
    if 'Join Filter' in json_node:
        conds.add(json_node['Join Filter'])
    if 'Merge Cond' in json_node:
        conds.add(json_node['Merge Cond'])
    # scan conds
    if 'Index Cond' in json_node:
        conds.add(json_node['Index Cond'])
    if 'Filter' in json_node:
        conds.add(json_node['Filter'])
    if 'Recheck Cond' in json_node:
        conds.add(json_node['Recheck Cond'])
    ####################################################################
    joins = []
    filters = []
    for cond in conds:
        res = condPipeline(cond, table)
        if res['joins'] is not None:
            joins.append(res['joins'])
        if len(res['filters']) > 0:
            filters.extend(res['filters'])
    if len(joins) == 0:
        return {
            'join' : 'NA',
            'filters' : filters
        }
    else:
        return {
            'join' : joins[0],
            'filters' : filters
        }

def condPipeline(string, table):
    ll_rep = nestedExpr('(',')').parseString(string).asList()
    flat_rep = flattenConds(ll_rep)
    return formatConds(flat_rep, table)

def flattenConds(rep): # naive flattening
    cons = ['AND','OR']
    ops = ['=','!=','<','>','<=','>=','<>']
    ress = []
    def dfss(rep):
        for r in rep:
            if r in cons:
                continue
            if isinstance(r, list) and len(r)>1 and r[1] in ops:
                ress.append(r)
            elif isinstance(r, list) and len(r)>1:
                dfss(r)
    dfss(rep)
    return ress        

def formatConds(conds, table): # either filter or join
    filters = []
    joins = None
    for cond in conds:
#         print(cond)
        if isinstance(cond[0], list):
#             print(cond[0], cond[2])
############## debugged for tpcds #
            try:
                cond[0] = cond[0][1][0]
            except:
                cond[0] = cond[0][-1][0]
###################################
            if not  isinstance(cond[2],list):
                return {'joins' : None, 'filters' : []}

        # if len(cond) == 3  and not cond[2].isnumeric(): # join #modified
        if len(cond) == 3 and not isinstance(cond[2],list) and not cond[2].isnumeric(): # join
            twoCols = cond[0], cond[2]
            twoCols = [table + '.' + col 
                  if len(col.split('.')) == 1 else col for col in twoCols] 
            joins = ' = '.join(sorted(twoCols))
                
        elif len(cond) > 3 and cond[3] in ['::timestamp', '::date']:
#             print(cond)
#             d = datetime.strptime(cond[2][1:11], '%Y-%m-%d')
            if cond[3] == '::timestamp':
                d = datetime.strptime(cond[2].split("'")[1], '%Y-%m-%d %H:%M:%S')
            else:
                d = datetime.strptime(cond[2].split("'")[1], '%Y-%m-%d')
            filt = [cond[0], cond[1], d.timestamp()]

            filters.append(filt)
            
        elif is_number(cond[2]):
            filt = [cond[0], cond[1], float(cond[2])]
            filters.append(filt)
        
        elif not isinstance(cond[2],list) and cond[2][1:-1].isnumeric(): # filter
            filt = [cond[0], cond[1], int(cond[2][1:-1])]
            filters.append(filt)
        elif not isinstance(cond[2],list) and cond[-1] != '::numeric': # string predicate
            ## do later
            ##TODO continue
            filt = [cond[0], cond[1], cond[2][1:-1]]
            filters.append(filt)
        
    for filt in filters:
        if len(filt[0].split('.'))==1:
            filt[0] = '.'.join((table, filt[0]))
#     print(filters, joins)
    return {
        'joins' : joins,
        'filters' : filters
    }

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


def flattenConds(rep):
    cons = ['AND','OR']
    ops = ['=','!=','<','>','<=','>=','<>']
    ress = []
    def dfss(rep):
        for r in rep:
            if r in cons:
                continue
            if isinstance(r, list) and len(r)>1 and r[1] in ops:
                ress.append(r)
            elif isinstance(r, list) and len(r)>1:
                dfss(r)
    dfss(rep)
    return ress