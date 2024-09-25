import asyncio
import os
from typing import *

import torch
import wandb
from typet5.data import get_tk_dataset_name
from typet5.function_dataset import data_project_from_dir
from typet5.model import ModelWrapper, ModelType
from typet5.train import TrainingConfig, PreprocessArgs
from typet5.type_env import AccuracyMetric
from typet5.utils import (
    PickleCache,
    assert_eq,
    get_dataroot,
    get_dataset_dir,
    get_eval_dir,
    get_gpu_id,
    get_model_dir,
    pickle_dump,
    pmap,
    pretty_print_dict,
    pretty_show_dict,
    proj_root,
    run_long_task,
    write_file,
)
from typet5.visualization import string_to_html
from termcolor import colored
from typet5.data import load_tokenized_srcsets, create_tokenized_srcsets

os.chdir(proj_root())

from typet5.function_decoding import (
    DecodingOrders,
    EvalResult,
    PreprocessArgs,
    RolloutCtx,
)
from typet5.function_dataset import sigmap_from_file_predictions
from typet5.static_analysis import SignatureErrorAnalysis
from typet5.experiments.typet5 import accs_as_table_row
from typet5.utils import decode_tokens, Path
from typet5.visualization import export_preds_on_code

def wandb_string(s: str):
    return wandb.Html(string_to_html(s))

def run():
    # experiment configurations
    quicktest = False

    gpu_id = get_gpu_id(1)
    # model_name = "model-v6--TrainingConfig(func_only=False, left_margin=2048, preamble_size=800, right_margin=1536)"
    model_name = "model-v6--TrainingConfig(func_only=False, imports_in_preamble=False, stub_in_preamble=False, left_margin=2048, right_margin=1536)"
    pre_args = PreprocessArgs(imports_in_preamble=False, stub_in_preamble=False)
    dataset_name = "ManyTypes4Py"
    # dataset_name = "InferTypes4Py"
    # dataset_name = "SPOT-src"
    experiment_name = dataset_name + ": " + model_name

    print(colored(f"Use GPU: {gpu_id}", "green"))

    # load test data
    sdata_name = get_tk_dataset_name(dataset_name, pre_args, func_only=False)
    sdata_path = get_dataroot() / "TokenizedSrcSets" / sdata_name
    recreate=False
    if recreate or not sdata_path.exists():
        create_tokenized_srcsets(
            dataset_name,
            sdata_path,
            func_only=False,
            pre_args=pre_args,
        )
    tk_dataset = load_tokenized_srcsets(
        sdata_path,
        quicktest=quicktest,
        sets_to_load=["test"],
    )

    # load model
    model = ModelWrapper.load_from_hub("MrVPlusOne/TypeT5-v7")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ctx_args = model.args.ctx_args
    model.args.sampling_max_tokens = ctx_args.ctx_size
    model.args.do_sample = False
    model.args.num_beams = 10
    model.args.tokens_per_type = 16

    eval_cache = PickleCache(get_eval_dir(dataset_name, model_name) / f"{pre_args}")
    # eval_cache.clear()
    pre_r = eval_cache.cached(
        "DatasetPredResult.pkl",
        lambda: model.eval_on_dataset(tk_dataset["test"]),
    )

    repos_dir = get_dataset_dir(dataset_name) / "repos" / "test"
    test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]
    test_projects = pmap(
        data_project_from_dir,
        test_repo_paths,
        desc="Loading test projects",
    )
    assert len(test_projects) > 0

    common_names = model.common_type_names
    # common_names = ModelWrapper.load_common_type_names(get_model_dir() / model_name)
    pred_map, label_map = sigmap_from_file_predictions(pre_r, test_projects, repos_dir)
    accs = {
        m.name: SignatureErrorAnalysis(pred_map, label_map, m).accuracies
        for m in AccuracyMetric.default_metrics(common_names)
    }

    from typet5.experiments.typet5 import accs_as_table_row
    accs_as_table_row(accs)
    pretty_print_dict(accs)

    export_to = Path(f"caches/model_predictions/eval_file_model/{dataset_name}")
    export_preds_on_code(pre_r.chunks, pre_r.predictions, export_to, AccuracyMetric(common_names))

if __name__ == "__main__":
    run()