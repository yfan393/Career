from db import *

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, sessionmaker
from tqdm import tqdm

import sys
import argparse
from pathlib import Path


def operation__dump_to_csv():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_path', '-o', required=True, default="Where the resultant CSV will be saved")
    ap.add_argument('--db_txn_batch_size', '-t', type=int, default=5000, help="Amount to fetch from each table in DB in one query.")
    args = ap.parse_args()

    output_path = Path(args.output_path)
    if not output_path or Path(output_path).is_dir():
        print("Invalid output path; aborting")
        exit(1)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print("Output CSV will be saved in: {}".format(output_path))

    db_classes = get_db_class('all')
    og_table_names = list(db_classes.keys())
    print("Found {} tables: {}".format(len(og_table_names), og_table_names))

    print("Initializing database")
    db_engine = get_db_engine()
    Session = sessionmaker(bind=db_engine)

    print("Finding member count")
    with Session() as sess:
        member_ids = [r[0] for r in sess.query(Member.member_id).all()]
    print("Found {} member ids".format(len(member_ids)))
    
    txn_batch = args.db_txn_batch_size
    colnames_all = None

    df_all = pd.DataFrame()

    print("Querying tables from DB with batch size of {}".format(txn_batch))
    # Build query
    with Session() as sess:
        q = sess.query(Member.member_id, Member.data_type)
        one2many_tables = ['QUALITY_DATA', 'humana_mays_target_member_visit_claims', 'humana_mays_target_member_conditions']
        for og_tbl_name in og_table_names:
            if og_tbl_name in one2many_tables:
                print("Skipping 1tomany tables: {}".format(og_tbl_name))
                continue
            db_cls = db_classes[og_tbl_name]
            cols = [c for c in db_cls.__table__.columns if c.name not in ('id', 'member_id')]
            if colnames_all is None:
                colnames_all = ['id', 'data_type']
            colnames_all.extend([c.name for c in cols])
            q = q.add_columns(*cols)
        for og_tbl_name in og_table_names:
            if og_tbl_name in one2many_tables:
                print("Skipping 1tomany tables: {}".format(og_tbl_name))
                continue
            db_cls = db_classes[og_tbl_name]
            #q = q.outerjoin(db_cls, Member.member_id == db_cls.member_id)
            q = q.join(db_cls, Member.member_id == db_cls.member_id)
        datas = []
        iterator = q.yield_per(txn_batch)
        for i, row in enumerate(prog := tqdm(iterator, total=len(member_ids))):
            datas.append(row)
        print("Creating DataFrame")
        df_all = pd.DataFrame(datas, columns=colnames_all)

    # for i in range(0, len(member_ids), txn_batch):
    #     ids_batch = member_ids[i:i+txn_batch]
    #     ids_batch_set = set(ids_batch)
    #     q = sess.query(Member.member_id, Member.data_type)
    #     for og_tbl_name in og_table_names:
    #         db_cls = db_classes[og_tbl_name]
    #         cols = [c for c in db_cls.__table__.columns if c.name not in ('id', 'member_id')]
    #         if i == 0:
    #             if colnames_all is None:
    #                 colnames_all = ['id', 'data_type']
    #             colnames_all.extend([c.name for c in cols])
    #         q = q.add_columns(*cols)

    #     for og_tbl_name in og_table_names:
    #         db_cls = db_classes[og_tbl_name]
    #         q = q.outerjoin(db_cls, Member.member_id == db_cls.member_id)
    #     q = q.filter(Member.member_id.in_(ids_batch_set))
    #     breakpoint()
    #     results = q.all()

    #     # Convert the results to a DataFrame
    #     df_batch = pd.DataFrame(results, columns=colnames_all)

    #     # Append the batch DataFrame to the main DataFrame
    #     df_all = pd.concat([df_all, df_batch], ignore_index=True)
        
    #     pbar.update(txn_batch)
    #     breakpoint()

    print("Saving as CSV ({})".format(output_path))
    df_all.to_csv(output_path, index=False)
    print("Done")


if __name__ == '__main__':
    sysargs_all = list(sys.argv)
    n_global_args = 2
    sys.argv = sysargs_all[:n_global_args]

    ops_list = ['dump_to_csv']

    ap_global = argparse.ArgumentParser()
    ap_global.add_argument('operation', choices=ops_list)
    args_global = ap_global.parse_args()
    op = args_global.operation

    sys.argv = [sysargs_all[0]] + sysargs_all[n_global_args:]
    globals()['operation__' + op]()