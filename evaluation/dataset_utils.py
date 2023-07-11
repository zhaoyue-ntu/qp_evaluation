import pandas as pd
import numpy as np
import json
from datetime import datetime
import time

import sys
sys.path.append('../evaluation/')

from feature_extractor import traversePlan, DatasetInfo

def get_col_min_max(minmax_df):
    col_min_max = {}
    for i, row in minmax_df.iterrows():
        if not 'name' in row:
            name = '.'.join((row['table'], row['column']))
        else:
            name = row['name']
        try:
            float(row['min'])
            col_min_max[name] = [float(row['min']),float(row['max'])]
        except:
            if len(row['min'])>15:
                d = datetime.strptime(row['min'], '%Y-%m-%d %H:%M:%S')
                try:
                    mi = d.timestamp()
                except:
                    mi = 0
                d = datetime.strptime(row['max'], '%Y-%m-%d %H:%M:%S')
                ma = d.timestamp()
            else:
                d = datetime.strptime(row['min'], '%Y-%m-%d')
                try:
                    mi = d.timestamp()
                except:
                    mi = 0
                d = datetime.strptime(row['max'], '%Y-%m-%d')
                ma = d.timestamp()
            col_min_max[name] = [mi, ma]
    return col_min_max

def get_costs(js_nodes): 
    costs = []
    for js_node in js_nodes:
        if 'Execution Time' in js_node:
            costs.append(js_node['Execution Time'])
        else:
            costs.append(js_node['Actual Total Time'])
    return costs

def get_index(df):
    idx_list = []
    for i, row in df.iterrows():
        if 'id' in row:
            qid = row['id']
            js_str = row['json']
        else:
            qid = i
            js_str = row['Plan_dump']
            
        if js_str == 'failed':
            continue
            
        for item in js_str.split(','):
            if item.startswith(' "Index Name"'):
                if item.split(':')[-1] not in idx_list:
                    item_part = item.split(':')[-1]
                    idx_list.append(item_part.split("\"")[-2])
                    
    return list(set(idx_list))

def df2nodes(df):
    t0 = time.time()
    idxs = []
    roots = []
    js_nodes = []
    for i, row in df.iterrows():
        if 'id' in row:
            idx = row['id']
            js_str = row['json']
        else:
            idx = i
            js_str = row['Plan_dump']
            
        if js_str == 'failed':
            continue
        js_node = json.loads(js_str)
        js_nodes.append(js_node)
        root = traversePlan(js_node)
        roots.append(root)
        idxs.append(idx)
    print('length: ', len(idxs), ', Time: ',  time.time()-t0)
    return roots, js_nodes, idxs

def get_imdb(dat_path):
    df = pd.DataFrame()
    for i in range(20):
        tmp_df = pd.read_csv(dat_path + 'plan_and_cost/train_plan_part{}.csv'.format(i))
        df = df.append(tmp_df)
    df.reset_index(drop=True, inplace=True)

    roots, js_nodes, idxs = df2nodes(df)
    index_list = get_index(df)
    costs = get_costs(js_nodes)

    minmax = pd.read_csv(dat_path + 'column_min_max_vals.csv')
    col_min_max = get_col_min_max(minmax)
    ds_info = DatasetInfo({})
    ds_info.construct_from_plans(roots)
    ds_info.get_columns(col_min_max)

    train_roots, val_roots = roots[:90000], roots[90000:]
    train_costs, val_costs = costs[:90000], costs[90000:]
    train_js_nodes, val_js_nodes = js_nodes[:90000], js_nodes[90000:]

    syn_df = pd.read_csv(dat_path+'synthetic_plan.csv')
    syn_roots, syn_js_nodes, _ = df2nodes(syn_df)
    syn_costs = get_costs(syn_js_nodes)

    job_light_df = pd.read_csv(dat_path+'job-light_plan.csv')
    job_light_roots, job_light_js_nodes, _ = df2nodes(job_light_df)
    job_light_costs = get_costs(job_light_js_nodes)

    return {
        'ds_info' : ds_info,
        'data_raw' : df,
        'indexes' : index_list,
        'col_min_max' : col_min_max,
        'total_roots' : roots,
        'total_costs' : costs,
        'train_roots' : train_roots,
        'train_costs' : train_costs,
        'train_js_nodes' : train_js_nodes,
        'val_roots' : val_roots,
        'val_costs' : val_costs,
        'val_js_nodes' : val_js_nodes,
        'syn_roots' : syn_roots,
        'syn_js_nodes' : syn_js_nodes,
        'syn_costs' : syn_costs,
        'job_light_roots' : job_light_roots,
        'job_light_js_nodes' : job_light_js_nodes,
        'job_light_costs' : job_light_costs,
    }

def get_tpch(dat_path):
    df = pd.read_csv(dat_path+'long_raw.csv')

    roots, js_nodes, idxs = df2nodes(df)
    index_list = get_index(df)
    costs = get_costs(js_nodes)

    minmax = pd.read_csv(dat_path + 'col_min_max.csv')
    col_min_max = get_col_min_max(minmax)
    ds_info = DatasetInfo({})
    ds_info.construct_from_plans(roots)
    ds_info.get_columns(col_min_max)

    ## split by run_id, TODO, split by query (template) id
    runs = df['Run_id'].unique()
    train_runs, val_runs, test_runs = np.split(runs,[35,45,],axis=0)
    train_ids = df.loc[df['Run_id'].isin(train_runs)].index.tolist()
    val_ids = df.loc[df['Run_id'].isin(val_runs)].index.tolist()
    test_ids = df.loc[df['Run_id'].isin(test_runs)].index.tolist()

    return {
        'ds_info' : ds_info,
        'data_raw' : df,
        'indexes' : index_list,
        'col_min_max' : col_min_max,
        'total_roots' : roots,
        'total_costs' : costs,
        'train_roots' : [roots[idx] for idx in train_ids],
        'train_costs' : [costs[idx] for idx in train_ids],
        'train_js_nodes' : [js_nodes[idx] for idx in train_ids],
        'val_roots' : [roots[idx] for idx in val_ids],
        'val_costs' : [costs[idx] for idx in val_ids],
        'val_js_nodes' : [js_nodes[idx] for idx in val_ids],
        'test_roots' : [roots[idx] for idx in test_ids],
        'test_js_nodes' : [js_nodes[idx] for idx in test_ids],
        'test_costs' : [costs[idx] for idx in test_ids],

        'train_ids' : train_ids,
        'val_ids' : val_ids,
        'test_ids' : test_ids,
    }


def get_tpcds(dat_path):
    df = pd.DataFrame()
    for i in range(21):
        tmp_df = pd.read_csv(dat_path+'long_raw_part{}.csv'.format(i))
        df = df.append(tmp_df)  
    index_list = get_index(df)
    df.reset_index(drop=True, inplace=True)
    df = df.loc[df['Index']=='[]'].reset_index(drop=True)

    roots, js_nodes, idxs = df2nodes(df)

    costs = get_costs(js_nodes)

    minmax = pd.read_csv(dat_path + 'col_min_max.csv')
    col_min_max = get_col_min_max(minmax)
    ds_info = DatasetInfo({})
    ds_info.construct_from_plans(roots)
    ds_info.get_columns(col_min_max)

    ## split by run_id, TODO, split by query (template) id
    runs = df['Run_id'].unique()
    train_runs, val_runs, test_runs = np.split(runs,[3,4,],axis=0)
    train_ids = df.loc[df['Run_id'].isin(train_runs)].index.tolist()
    val_ids = df.loc[df['Run_id'].isin(val_runs)].index.tolist()
    test_ids = df.loc[df['Run_id'].isin(test_runs)].index.tolist()

    return {
        'ds_info' : ds_info,
        'data_raw' : df,
        'indexes' : index_list,
        'col_min_max' : col_min_max,
        'total_roots' : roots,
        'total_costs' : costs,
        'train_roots' : [roots[idx] for idx in train_ids],
        'train_costs' : [costs[idx] for idx in train_ids],
        'train_js_nodes' : [js_nodes[idx] for idx in train_ids],
        'val_roots' : [roots[idx] for idx in val_ids],
        'val_costs' : [costs[idx] for idx in val_ids],
        'val_js_nodes' : [js_nodes[idx] for idx in val_ids],
        'test_roots' : [roots[idx] for idx in test_ids],
        'test_js_nodes' : [js_nodes[idx] for idx in test_ids],
        'test_costs' : [costs[idx] for idx in test_ids],

        'train_ids' : train_ids,
        'val_ids' : val_ids,
        'test_ids' : test_ids,
    }


def get_stats(dat_path):
    df = pd.DataFrame()
    for i in range(71):
        tmp_df = pd.read_csv(dat_path + 'trial0/train_plan_chunk{}_run0.csv'.format(i))
        df = df.append(tmp_df)
    df.reset_index(drop=True, inplace=True)
    df = df.sample(frac=1, random_state=2).reset_index(drop=True)
    
    roots, js_nodes, idxs = df2nodes(df)
    index_list = get_index(df)
    costs = get_costs(js_nodes)

    minmax = pd.read_csv(dat_path + 'column_min_max_vals.csv')
    col_min_max = get_col_min_max(minmax)
    ds_info = DatasetInfo({})
    ds_info.construct_from_plans(roots)
    ds_info.get_columns(col_min_max)

    train_roots = roots
    train_costs = costs
    train_js_nodes = js_nodes
    
    df = pd.read_csv(dat_path+'trial0/test_plan__run0.csv')
    val_roots, val_js_nodes, _ = df2nodes(df)
    val_costs = get_costs(val_js_nodes)
    
    df = pd.read_csv(dat_path+'trial0/workload_plan_run0.csv')
    test_roots, test_js_nodes, _ = df2nodes(df)
    test_costs = get_costs(test_js_nodes)

    return {
        'ds_info' : ds_info,
        'data_raw' : df,
        'indexes' : index_list,
        'col_min_max' : col_min_max,
        'total_roots' : roots,
        'total_costs' : costs,
        'train_roots' : train_roots,
        'train_costs' : train_costs,
        'train_js_nodes' : train_js_nodes,
        'val_roots' : val_roots,
        'val_costs' : val_costs,
        'val_js_nodes' : val_js_nodes,
        'test_roots' : test_roots,
        'test_js_nodes' : test_js_nodes,
        'test_costs' : test_costs
    }

stats_schema = {
    'REL_NAMES': [
    'badges', 
    'comments', 
    'posthistory', 
    'postlinks', 
    'posts', 
    'tags',
    'users',
    'votes'],
    'REL_ATTR_LIST_DICT' : {'badges': ['Id', 'UserId', 'Date'],
 'comments': ['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
 'posthistory': ['Id',
  'PostHistoryTypeId',
  'PostId',
  'CreationDate',
  'UserId'],
 'postlinks': ['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
 'posts': ['Id',
  'PostTypeId',
  'CreationDate',
  'Score',
  'ViewCount',
  'OwnerUserId',
  'AnswerCount',
  'CommentCount',
  'FavoriteCount',
  'LastEditorUserId'],
 'tags': ['Id', 'Count', 'ExcerptPostId'],
 'users': ['Id',
  'Reputation',
  'CreationDate',
  'Views',
  'UpVotes',
  'DownVotes'],
 'votes': ['Id',
  'PostId',
  'VoteTypeId',
  'CreationDate',
  'UserId',
  'BountyAmount']}}


imdb_schema = {
    'REL_NAMES': [
        'title', 
        'movie_companies', 
        'cast_info', 
        'movie_info_idx', 
        'movie_keyword', 
        'movie_info'
    ],
    'REL_ATTR_LIST_DICT' : {
        'title':
            ['t.id',
             't.kind_id',
             't.production_year'],

        'movie_companies':
            ['mc.id',
             'mc.company_id',
             'mc.movie_id',
             'mc.company_type_id'],

        'cast_info':
            ['ci.id',
             'ci.movie_id',
             'ci.person_id',
             'ci.role_id'],

        'movie_info_idx':
            ['mi_idx.id',
             'mi_idx.movie_id',
             'mi_idx.info_type_id'],

        'movie_info':
            ['mi.id',
             'mi.movie_id',
             'mi.info_type_id'],

        'movie_keyword':
            ['mk.id',
             'mk.movie_id',
             'mk.keyword_id']
    }
}


tpch_schema = {
    'REL_NAMES' : [
        'customer', 
        'lineitem', 
        'nation', 
        'orders', 
        'part', 
        'partsupp', 
        'region', 
        'supplier'
    ],
    'REL_ATTR_LIST_DICT' : {
        'customer':
            ['c_custkey',
             'c_name',
             'c_address',
             'c_nationkey',
             'c_phone',
             'c_acctbal',
             'c_mktsegment',
             'c_comment'],
        'lineitem':
            ['l_orderkey',
             'l_partkey',
             'l_suppkey',
             'l_linenumber',
             'l_quantity',
             'l_extendedprice',
             'l_discount',
             'l_tax',
             'l_returnflag',
             'l_linestatus',
             'l_shipdate',
             'l_commitdate',
             'l_receiptdate',
             'l_shipinstruct',
             'l_shipmode',
             'l_comment'],
        'nation':
            ['n_nationkey',
             'n_name',
             'n_regionkey',
             'n_comment'],
        'orders':
            ['o_orderkey',
             'o_custkey',
             'o_orderstatus',
             'o_totalprice',
             'o_orderdate',
             'o_orderpriority',
             'o_clerk',
             'o_shippriority',
             'o_comment'],
        'part':
            ['p_partkey',
             'p_name',
             'p_mfgr',
             'p_brand',
             'p_type',
             'p_size',
             'p_container',
             'p_retailprice',
             'p_comment'],
        'partsupp':
            ['ps_partkey',
             'ps_suppkey',
             'ps_availqty',
             'ps_supplycost',
             'ps_comment'],
        'region':
            ['r_regionkey',
             'r_name',
             'r_comment'],
        'supplier':
            ['s_suppkey',
             's_name',
             's_address',
             's_nationkey',
             's_phone',
             's_acctbal',
             's_comment']
    }
}


tpcds_schema = {
    'REL_NAMES' : ['store_sales', 'store_returns', 'catalog_sales', 'catalog_returns', 'web_sales', 'web_returns', 
        'inventory', 'store', 'call_center', 'catalog_page', 'web_site',  'web_page', 'warehouse', 
        'customer', 'customer_address', 'customer_demographics', 'date_dim', 'household_demographics', 
        'item', 'income_band', 'promotion', 'reason', 'ship_mode', 'time_dim'],
    'REL_ATTR_LIST_DICT' : {
        'store_sales':
            ['ss_sold_date_sk',
             'ss_sold_time_sk',
             'ss_item_sk',
             'ss_customer_sk',
             'ss_cdemo_sk',
             'ss_hdemo_sk',
             'ss_addr_sk',
             'ss_store_sk',
             'ss_promo_sk',
             'ss_ticket_number',
             'ss_quantity',
             'ss_wholesale_cost',
             'ss_list_price',
             'ss_sales_price',
             'ss_ext_discount_amt',
             'ss_ext_sales_price',
             'ss_ext_wholesale_cost',
             'ss_ext_list_price',
             'ss_ext_tax',
             'ss_coupon_amt',
             'ss_net_paid',
             'ss_net_paid_inc_tax',
             'ss_net_profit'],

        'store_returns':
            ['sr_returned_date_sk',
             'sr_return_time_sk',
             'sr_item_sk',
             'sr_customer_sk',
             'sr_cdemo_sk',
             'sr_hdemo_sk',
             'sr_addr_sk',
             'sr_store_sk',
             'sr_reason_sk',
             'sr_ticket_number',
             'sr_return_quantity',
             'sr_return_amt',
             'sr_return_tax',
             'sr_return_amt_inc_tax',
             'sr_fee',
             'sr_return_ship_cost',
             'sr_refunded_cash',
             'sr_reversed_charge',
             'sr_store_credit',
             'sr_net_loss'],

        'catalog_sales':
            ['cs_sold_date_sk',
             'cs_sold_time_sk',
             'cs_ship_date_sk',
             'cs_bill_customer_sk',
             'cs_bill_cdemo_sk',
             'cs_bill_hdemo_sk',
             'cs_bill_addr_sk',
             'cs_ship_customer_sk',
             'cs_ship_cdemo_sk',
             'cs_ship_hdemo_sk',
             'cs_ship_addr_sk',
             'cs_call_center_sk',
             'cs_catalog_page_sk',
             'cs_ship_mode_sk',
             'cs_warehouse_sk',
             'cs_item_sk',
             'cs_promo_sk',
             'cs_order_number',
             'cs_quantity',
             'cs_wholesale_cost',
             'cs_list_price',
             'cs_sales_price',
             'cs_ext_discount_amt',
             'cs_ext_sales_price',
             'cs_ext_wholesale_cost',
             'cs_ext_list_price',
             'cs_ext_tax',
             'cs_coupon_amt',
             'cs_ext_ship_cost',
             'cs_net_paid',
             'cs_net_paid_inc_tax',
             'cs_net_paid_inc_ship',
             'cs_net_paid_inc_ship_tax',
             'cs_net_profit'],

        'catalog_returns':
            ['cr_returned_date_sk',
             'cr_returned_time_sk',
             'cr_item_sk',
             'cr_refunded_customer_sk',
             'cr_refunded_cdemo_sk',
             'cr_refunded_hdemo_sk',
             'cr_refunded_addr_sk',
             'cr_returning_customer_sk',
             'cr_returning_cdemo_sk',
             'cr_returning_hdemo_sk',
             'cr_returning_addr_sk',
             'cr_call_center_sk',
             'cr_catalog_page_sk',
             'cr_ship_mode_sk',
             'cr_warehouse_sk',
             'cr_reason_sk',
             'cr_order_number',
             'cr_return_quantity',
             'cr_return_amount',
             'cr_return_tax',
             'cr_return_amt_inc_tax',
             'cr_fee',
             'cr_return_ship_cost',
             'cr_refunded_cash',
             'cr_reversed_charge',
             'cr_store_credit',
             'cr_net_loss'],

        'web_sales':
            ['ws_sold_date_sk',
             'ws_sold_time_sk',
             'ws_ship_date_sk',
             'ws_item_sk',
             'ws_bill_customer_sk',
             'ws_bill_cdemo_sk',
             'ws_bill_hdemo_sk',
             'ws_bill_addr_sk',
             'ws_ship_customer_sk',
             'ws_ship_cdemo_sk',
             'ws_ship_hdemo_sk',
             'ws_ship_addr_sk',
             'ws_web_page_sk',
             'ws_web_site_sk',
             'ws_ship_mode_sk',
             'ws_warehouse_sk',
             'ws_promo_sk',
             'ws_order_number',
             'ws_quantity',
             'ws_wholesale_cost',
             'ws_list_price',
             'ws_sales_price',
             'ws_ext_discount_amt',
             'ws_ext_sales_price',
             'ws_ext_wholesale_cost',
             'ws_ext_list_price',
             'ws_ext_tax',
             'ws_coupon_amt',
             'ws_ext_ship_cost',
             'ws_net_paid',
             'ws_net_paid_inc_tax',
             'ws_net_paid_inc_ship',
             'ws_net_paid_inc_ship_tax',
             'ws_net_profit'],

        'web_returns':
            ['wr_returned_date_sk',
             'wr_returned_time_sk',
             'wr_item_sk',
             'wr_refunded_customer_sk',
             'wr_refunded_cdemo_sk',
             'wr_refunded_hdemo_sk',
             'wr_refunded_addr_sk',
             'wr_returning_customer_sk',
             'wr_returning_cdemo_sk',
             'wr_returning_hdemo_sk',
             'wr_returning_addr_sk',
             'wr_web_page_sk',
             'wr_reason_sk',
             'wr_order_number',
             'wr_return_quantity',
             'wr_return_amount',
             'wr_return_tax',
             'wr_return_amt_inc_tax',
             'wr_fee',
             'wr_return_ship_cost',
             'wr_refunded_cash',
             'wr_reversed_charge',
             'wr_account_credit',
             'wr_net_loss'],

        'inventory':
            ['inv_date_sk',
             'inv_item_sk',
             'inv_warehouse_sk',
             'inv_quantity_on_hand'],

        'store':
            ['s_store_sk',
             's_store_id',
             's_rec_start_date',
             's_rec_end_date',
             's_closed_date_sk',
             's_store_name',
             's_number_employees',
             's_floor_space',
             's_hours',
             'S_manager',
             'S_market_id',
             'S_geography_class',
             'S_market_desc',
             's_market_manager',
             's_division_id',
             's_division_name',
             's_company_id',
             's_company_name',
             's_street_number',
             's_street_name',
             's_street_type',
             's_suite_number',
             's_city',
             's_county',
             's_state',
             's_zip',
             's_country',
             's_gmt_offset',
             's_tax_percentage'],

        'call_center':
            ['cc_call_center_sk',
             'cc_call_center_id',
             'cc_rec_start_date',
             'cc_rec_end_date',
             'cc_closed_date_sk',
             'cc_open_date_sk',
             'cc_name',
             'cc_class',
             'cc_employees',
             'cc_sq_ft',
             'cc_hours',
             'cc_manager',
             'cc_mkt_id',
             'cc_mkt_class',
             'cc_mkt_desc',
             'cc_market_manager',
             'cc_division',
             'cc_division_name',
             'cc_company',
             'cc_company_name',
             'cc_street_number',
             'cc_street_name',
             'cc_street_type',
             'cc_suite_number',
             'cc_city',
             'cc_county',
             'cc_state',
             'cc_zip',
             'cc_country',
             'cc_gmt_offset',
             'cc_tax_percentage'],

        'catalog_page':
            ['cp_catalog_page_sk',
             'cp_catalog_page_id',
             'cp_start_date_sk',
             'cp_end_date_sk',
             'cp_department',
             'cp_catalog_number',
             'cp_catalog_page_number',
             'cp_description',
             'cp_type'],

        'web_site':
            ['web_site_sk',
             'web_site_id',
             'web_rec_start_date',
             'web_rec_end_date',
             'web_name',
             'web_open_date_sk',
             'web_close_date_sk',
             'web_class',
             'web_manager',
             'web_mkt_id',
             'web_mkt_class',
             'web_mkt_desc',
             'web_market_manager',
             'web_company_id',
             'web_company_name',
             'web_street_number',
             'web_street_name',
             'web_street_type',
             'web_suite_number',
             'web_city',
             'web_county',
             'web_state',
             'web_zip',
             'web_country',
             'web_gmt_offset',
             'web_tax_percentage'],

        'web_page':
            ['wp_web_page_sk',
             'wp_web_page_id',
             'wp_rec_start_date',
             'wp_rec_end_date',
             'wp_creation_date_sk',
             'wp_access_date_sk',
             'wp_autogen_flag',
             'wp_customer_sk',
             'wp_url',
             'wp_type',
             'wp_char_count',
             'wp_link_count',
             'wp_image_count',
             'wp_max_ad_count'],

        'warehouse':
            ['w_warehouse_sk',
             'w_warehouse_id',
             'w_warehouse_name',
             'w_warehouse_sq_ft',
             'w_street_number',
             'w_street_name',
             'w_street_type',
             'w_suite_number',
             'w_city',
             'w_county',
             'w_state',
             'w_zip',
             'w_country',
             'w_gmt_offset'],

        'customer':
            ['c_customer_sk',
             'c_customer_id',
             'c_current_cdemo_sk',
             'c_current_hdemo_sk',
             'c_current_addr_sk',
             'c_first_shipto_date_sk',
             'c_first_sales_date_sk',
             'c_salutation',
             'c_first_name',
             'c_last_name',
             'c_preferred_cust_flag',
             'c_birth_day',
             'c_birth_month',
             'c_birth_year',
             'c_birth_country',
             'c_login',
             'c_email_address',
             'c_last_review_date_sk'],

        'customer_address':
            ['ca_address_sk',
             'ca_address_id',
             'ca_street_number',
             'ca_street_name',
             'ca_street_type',
             'ca_suite_number',
             'ca_city',
             'ca_county',
             'ca_state',
             'ca_zip',
             'ca_country',
             'ca_gmt_offset',
             'ca_location_type'],

        'customer_demographics':
            ['cd_demo_sk',
             'cd_gender',
             'cd_marital_status',
             'cd_education_status',
             'cd_purchase_estimate',
             'cd_credit_rating',
             'cd_dep_count',
             'cd_dep_employed_count',
             'cd_dep_college_count'],

        'date_dim':
            ['d_date_sk',
             'd_date_id',
             'd_date',
             'd_month_seq',
             'd_week_seq',
             'd_quarter_seq',
             'd_year',
             'd_dow',
             'd_moy',
             'd_dom',
             'd_qoy',
             'd_fy_year',
             'd_fy_quarter_seq',
             'd_fy_week_seq',
             'd_day_name',
             'd_quarter_name',
             'd_holiday',
             'd_weekend',
             'd_following_holiday',
             'd_first_dom',
             'd_last_dom',
             'd_same_day_ly',
             'd_same_day_lq',
             'd_current_day',
             'd_current_week',
             'd_current_month',
             'd_current_quarter',
             'd_current_year'],

        'household_demographics':
            ['hd_demo_sk',
             'hd_income_band_sk',
             'hd_buy_potential',
             'hd_dep_count',
             'hd_vehicle_count'],

        'item':
            ['i_item_sk',
             'i_item_id',
             'i_rec_start_date',
             'i_rec_end_date',
             'i_item_desc',
             'i_current_price',
             'i_wholesale_cost',
             'i_brand_id',
             'i_brand',
             'i_class_id',
             'i_class',
             'i_category_id',
             'i_category',
             'i_manufact_id',
             'i_manufact',
             'i_size',
             'i_formulation',
             'i_color',
             'i_units',
             'i_container',
             'i_manager_id',
             'i_product_name'],

        'income_band':
            ['ib_income_band_sk',
             'ib_lower_bound',
             'ib_upper_bound'],

        'promotion':
            ['p_promo_sk',
             'p_promo_id',
             'p_start_date_sk',
             'p_end_date_sk',
             'p_item_sk',
             'p_cost',
             'p_response_target',
             'p_promo_name',
             'p_channel_dmail',
             'p_channel_email',
             'p_channel_catalog',
             'p_channel_tv',
             'p_channel_radio',
             'p_channel_press',
             'p_channel_event',
             'p_channel_demo',
             'p_channel_details',
             'p_purpose',
             'p_discount_active'],

        'reason':
            ['r_reason_sk',
             'r_reason_id',
             'r_reason_desc'],

        'ship_mode':
            ['sm_ship_mode_sk',
             'sm_ship_mode_id',
             'sm_type',
             'sm_code',
             'sm_carrier',
             'sm_contract'],

        'time_dim':
            ['t_time_sk',
             't_time_id',
             't_time',
             't_hour',
             't_minute',
             't_second',
             't_am_pm',
             't_shift',
             't_sub_shift'
             't_meal_time']

    }

}
