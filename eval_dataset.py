import asyncio
import os
from typing import *
<<<<<<< HEAD
import copy
import torch
import wandb
from typet5.experiments.typet5 import TypeT5Configs
=======

import torch
import wandb
>>>>>>> e881c16262bd72fa165cb7db5389b153212086ed
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
<<<<<<< HEAD
    get_modified_args,
    pickle_load,
=======
>>>>>>> e881c16262bd72fa165cb7db5389b153212086ed
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
<<<<<<< HEAD
from typet5.experiments.typet5 import accs_as_table_row
from typet5.function_decoding import DecodingOrders, EvalResult, RolloutCtx
=======
>>>>>>> e881c16262bd72fa165cb7db5389b153212086ed

def wandb_string(s: str):
    return wandb.Html(string_to_html(s))

def run():
    # experiment configurations
    quicktest = False

<<<<<<< HEAD
    load_results = False
    use_oracle = False
    train_config = TypeT5Configs.Default
    model_name = train_config.get_model_name()

    gpu_id = get_gpu_id(0)
    # model_name = "model-v6--TrainingConfig(func_only=False, left_margin=2048, preamble_size=800, right_margin=1536)"
    test_pre_args = train_config.pre_args
    dataset_name = "ManyTypes4Py"
    oracle_tag = "(use-oracle) " if use_oracle else ""
    # group_tag = "(implicit_imports, new) "
    # group_tag = "(ablation) "
    group_tag = ""
    experiment_name = oracle_tag + group_tag + model_name


    print(colored(f"Use GPU: {gpu_id}", "green"))
    model = ModelWrapper.load_from_hub("MrVPlusOne/TypeT5-v7")

    # print(model.args.top_k)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded:", model_name)

    ctx_args = model.args.ctx_args
    ctx_args.max_labels = 16
    model.args.sampling_max_tokens = ctx_args.ctx_size
    model.args.do_sample = False
    model.args.num_beams = 16
    model.args.tokens_per_type = 16
    model.args.num_beam_groups = 1

    print(model.args.num_beam_groups)
    print(model.args)

    repos_dir = get_dataset_dir(dataset_name) / "repos" / "valid"
    test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]

    # test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir() and "id3c" in f.name]
    # print(test_repo_paths)

=======
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

    print(f"Loading tokenized srcsets from {sdata_name}")
    exit()

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
>>>>>>> e881c16262bd72fa165cb7db5389b153212086ed
    test_projects = pmap(
        data_project_from_dir,
        test_repo_paths,
        desc="Loading test projects",
    )
    assert len(test_projects) > 0

<<<<<<< HEAD
    rctx = RolloutCtx(model=model)

    decode_orders = {
        "double-traversal": DecodingOrders.DoubleTraversal(),
        # "reverse-double-traversal": DecodingOrders.Reversed(
        #     DecodingOrders.DoubleTraversal()
        # ),
        # "non-incr": DecodingOrders.IndependentOrder(),
        # "random": DecodingOrders.RandomOrder(),
        # "no-neighbors": DecodingOrders.IndependentOrder(),
        # "callee2caller": DecodingOrders.Callee2Caller(),
        # "caller2callee": DecodingOrders.Caller2Callee(),
        # "random-twice": DecodingOrders.RandomTwice(),
    }

    metrics = AccuracyMetric.default_metrics(model.common_type_names)
    with run_long_task("Evaluating different decoding strategy", notify=not load_results):
        results_dir = get_eval_dir(dataset_name, experiment_name)
        results_dir.mkdir(exist_ok=True, parents=True)
        print(colored(f"Results will be saved to: {str(results_dir)}", "green"))

        # if not load_results:
        #     wandb.init(
        #         project="SPOT-eval",
        #         name=dataset_name + ": " + experiment_name,
        #         dir=str(results_dir),
        #         config=get_modified_args(model.args),
        #     )

        evals = dict[str, EvalResult]()
        for oname, order in decode_orders.items():
            result_path = results_dir / f"{oname}-EvalResultAllTest_Valid_0402.pkl"
            if not load_results:
                print(f"Evaluating decoding strategy: {oname}")
                pre_args = copy.deepcopy(test_pre_args)
                if oname == "no-neighbors":
                    pre_args.max_callers = 0
                    pre_args.max_callees = 0
                evalr = asyncio.run(
                    rctx.evaluate_on_projects(
                        test_projects,  # type: ignore
                        pre_args,
                        order,
                        use_oracle=use_oracle,
                    )
                )
                pickle_dump(result_path, evalr)
            else:
                if not result_path.exists():
                    print(f"Result file not found, skip: {result_path}")
                    continue
                evalr = pickle_load(result_path)
            evals[oname] = evalr
            accs = {m.name: evalr.error_analysis(None, m).accuracies for m in metrics}
            accs_str = pretty_show_dict(accs)
            write_file(results_dir / f"{oname}-accuracy.txt", accs_str)
            # if not load_results:
            #     wandb.log({f"test/{oname}": wandb_string(accs_str)})
            print(f"========== {oname} ===========")
            print(accs_str)
            accs_as_table_row(accs)

    # %%
    if False:
        # print predictions
        for oname, evalr in evals.items():
            print(f"========== {oname} ===========")
            evalr.print_predictions()

    import prettytable as pt

    # %%
    from prettytable import PrettyTable

    common_type_names = ModelWrapper.load_common_type_names(get_model_dir() / model_name)
    results_table = PrettyTable()
    results_table.field_names = ["order", *(m.name for m in metrics)]
    results_table.align = "r"
    results_table.set_style(pt.SINGLE_BORDER)
    results_table.float_format = ".4"

    for oname in evals:
        accs = [
            evals[oname].error_analysis(None, metric).accuracies[metric.name].acc
            for metric in metrics
        ]
        results_table.add_row([oname, *accs])

    print(results_table)
    # write_file(results_dir / "comparison.txt", results_table.get_string())
=======
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
>>>>>>> e881c16262bd72fa165cb7db5389b153212086ed

if __name__ == "__main__":
    run()