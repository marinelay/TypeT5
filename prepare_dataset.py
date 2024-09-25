import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import libcst as cst
from tqdm import tqdm

from typet5.data import GitRepo
from typet5.type_env import collect_annots_info, mypy_checker
from typet5.utils import proj_root, read_file, write_file

os.chdir(proj_root())

import pickle

from typet5 import proj_root
from typet5.data import get_dataset_dir, get_tk_dataset_name, PreprocessArgs
import typet5.function_dataset as fd
from typet5.utils import Path, run_long_task, DefaultTokenizer, not_none
from typet5.data import create_tokenized_srcsets, load_tokenized_srcsets
import subprocess

import plotly.express as px
from pandas import DataFrame

from typet5.utils import cumulative_counts

def run():
    dataset_name = "ManyTypes4Py"
    # repos_split_path = proj_root() /  "data/repos_split.pkl"
    repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"

    recreate = False
    func_only = True # whether to create functional data (for TypeT5) or chunk data (for CodeT5)
    pre_args = PreprocessArgs()
    data_reduction = 1

    tk_src_name = get_tk_dataset_name(
        dataset_name, pre_args, func_only, data_reduction=data_reduction,
    )
    print(tk_src_name)
    datasets_path = get_data_dir() / "SPOT-data" / tk_src_name
    if recreate or not datasets_path.exists():
        create_tokenized_srcsets(
            proj_root() / "data/repos_split.pkl",
            datasets_path,
            func_only=func_only,
            pre_args=pre_args,
            data_reduction=data_reduction,
        )
    tk_dataset = load_tokenized_srcsets(
        datadir,
        tk_src_name,
    )

    len_counts = [len(src.tokenized_code) for src in tk_dataset["train"].all_srcs]
    xs, ys = cumulative_counts(len_counts)
    px.line(
        DataFrame({"tokens_per_file": xs, "n_files": ys}), x="tokens_per_file", y="n_files"
    )
    print("dataset:", datasets_path)
    tk_dataset["train"].print_stats()

if __name__ == "__main__":
    run()