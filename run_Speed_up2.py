#!/bin/env python3
# -* coding = UTF-8 *-
# Changed by Rock Nhu (Shixuan Huang)
# TODO: more robust and easier to read

import os
import re
import argparse
from Bio import SeqIO
import pandas as pd
import subprocess


def get_args():
    '''Arguments parser setting'''
    parser = argparse.ArgumentParser(
        description='Run total protocol of PhaGCN2. This script could run PhaGCN2 in other directory.')
    parser.add_argument('-i', '--contigs', required=True, type=str,
                        help='The input contigs in fasta format')
    parser.add_argument('--phagcn_dir', required=True, type=str,
                        help='The phagcn dir which have phagcn2 python scripts')
    parser.add_argument('--database_dir', required=True, type=str,
                        help='The database dir which have ALL_protein.fasta and database.dmnd')
    parser.add_argument('-l', '--len', required=False, type=int, default=8000,
                        help='Length of the lowest contig')
    parser.add_argument('-t', '--threads', required=False, type=int, default=128,
                        help='The threads of diamond')
    return parser.parse_args()


def run_cmd(cmd, error_log=None, output_log=None, error_cmd=None):
    '''Shell command runner'''
    try:
        print(output_log, flush=True)
        out = subprocess.check_call(cmd, shell=True)
        return out or True
    except Exception:
        if error_cmd:
            subprocess.check_call(error_cmd, shell=True)
        print(error_log, flush=True)
        return False


def exit_by_hand(state, info):
    '''The exit'''
    if not state:
        print(info)
        exit(1)


def check_folder(file_name):
    '''Make directory'''
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    else:
        check = run_cmd(cmd=f'rm -rf {file_name}\nmkdir -p {file_name}',
                        error_log="Cannot clean your folder... permission denied",
                        output_log=f"folder {file_name} exist... cleaning dictionary")
        if not check:
            exit(1)


def special_match(strg, search=re.compile(r'[^ACGT]').search):
    '''Check odd character'''
    return not bool(search(strg))


def env_setting():
    '''Make temporary directory'''
    check_folder("self_database")  # to save self diamond database
    check_folder("input")
    check_folder("pred")
    check_folder("Split_files")
    check_folder("network")


def dmnd_db_setting(args):
    '''Making the self-diamond database'''
    db_state = run_cmd(f'diamond makedb --threads {args.threads} --in {args.database}/ALL_protein.fasta -d {args.database}/database.dmnd',
                       "Creating Diamond database...")
    exit_by_hand(db_state, "create database failed")
    dmnd_state = run_cmd(f'diamond blastp --threads {args.threads} --sensitive -d {args.database}/database.dmnd -q {args.database}/ALL_protein.fasta -o self_database/database.self-diamond.tab',
                         "Running Diamond...")
    exit_by_hand(dmnd_state, "create database failed")
    diamond_out_fp = "self_database/database.self-diamond.tab"
    database_abc_fp = "self_database/database.self-diamond.tab.abc"
    awk_state = run_cmd(
        f"awk '$1!=$2 {{print $1,$2,$11}}' {diamond_out_fp} > {database_abc_fp}")
    exit_by_hand(awk_state, "create database failed")


def split_fa(args):
    '''Split contigs to each one, the file_id is indexes of records'''
    records = []  # the spoon to scoop contigs soup
    count = 0  # count the contig to scoop
    file_id = 0  # how many scoops in this soup
    for record in SeqIO.parse(args.contigs, 'fasta'):
        if count != 0 and count % 1000 == 0:  # every spoon only take 1000 contigs
            SeqIO.write(
                records, f"Split_files/contig_{file_id}.fasta", "fasta")
            # clean the spoon
            records = []
            count = 0
            # count the scoop number
            file_id += 1
        # only take upper character in contig
        seq = str(record.seq).upper()
        if special_match(seq) and len(record.seq) > args.len:  # the contig is ok
            records.append(record)
            count += 1
    # A little soup left in the spoon, also scoop it out
    SeqIO.write(records, f"Split_files/contig_{file_id}.fasta", "fasta")
    file_id += 1
    return file_id


def _protocol(file_id):
    # each one run protocol
    for i in range(file_id):
        out = run_cmd(f"mv Split_files/contig_{i}.fasta input/",
                      f"Moving file Error for file contig_{i}")
        if not out:
            continue

        out = run_cmd(cmd="python run_CNN.py",
                      error_log=f"Pre-trained CNN Error for file contig_{i}",
                      error_cmd="rm input/*")
        if not out:
            continue

        out = run_cmd(cmd=f"python run_KnowledgeGraph.py --n {i}",
                      error_log=f"Knowledge Graph Error for file contig_{i}",
                      error_cmd="rm input/*")
        if not out:
            continue

        out = run_cmd(cmd="python run_GCN.py",
                      error_log=f"GCN Error for file contig_{i}",
                      error_cmd="rm input/*")
        if not out:
            continue

        # Clean temp files
        run_cmd("rm input/*")

        name_list = pd.read_csv("name_list.csv")
        prediction = pd.read_csv("prediction.csv")
        prediction = prediction.rename(columns={'contig_names': 'idx'})
        contig_to_pred = pd.merge(name_list, prediction, on='idx')
        contig_to_pred.to_csv(f"pred/contig_{str(i)}.csv", index=None)

        run_cmd("rm name_list.csv prediction.csv")

    run_cmd("cat pred/* > final_prediction.csv")


def main():
    args = get_args()
    env_setting()
    dmnd_db_setting(args)
    split_fa(args)
    _protocol(args)


if __name__ == '__main':
    main()
