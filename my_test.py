# %%
import os
from typing import *

import torch
import asyncio
import json
import signal
import time
import libcst

from time import monotonic

from typet5.model import ModelWrapper
from typet5.train import PreprocessArgs
from typet5.utils import *
from typet5.function_decoding import (
    RolloutCtx,
    PreprocessArgs,
    DecodingOrders,
    AccuracyMetric,
)
from typet5.static_analysis import PythonProject, FunctionSignature, VariableSignature

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

def alarm_handler(signum, frame):
    print("Time is up!")
    raise TimeOutException()

target_projects = [
    # "airflow-3831",
    # "airflow-4674",
    # "airflow-5686",
    # "airflow-6036",
    # "airflow-8151",
    # # "airflow-14513",
    # "airflow-14686",
    # "beets-3360",
    # "core-8065",
    # "core-21734",
    # "core-29829",
    # "core-32222",
    # "core-32318",
    # "core-40034",
    # "kivy-6954",
    # "luigi-1836",
    # "pandas-17609",
    # "pandas-21540",
    # "pandas-22378",
    # "pandas-22804",
    # "pandas-24572",
    # "pandas-28412",
    # # "pandas-30532",
    # "pandas-36950",
    # "pandas-37547",
    # "pandas-38431",
    # "pandas-39028-1",
    # "pandas-41915",
    # # "rasa-8704",
    # "requests-3179",
    # "requests-3390",
    # "requests-4723",
    # "salt-33908",
    # "salt-38947",
    # "salt-52624",
    # "salt-53394",
    # "salt-54240",
    # "salt-54785",
    # "salt-56381",
    # "sanic-1334",
    # # "sanic-2008",
    # "scikitlearn-7259",
    # "scikitlearn-8973",
    # "scikitlearn-12603",
    # "Zappa-388",
    # "ansible-1",
    # "keras-34",
    # "keras-39",
    # "luigi-4",
    # "luigi-14",
    # "pandas-49",
    # "pandas-57",
    # "pandas-158",
    # "scrapy-1",
    # "scrapy-2",
    # "spacy-5",
    # # "tqdm-3",
    # #"youtubedl-11",
    # #"youtubedl-16",
    # "matplotlib-3",
    # "matplotlib-7",
    # "matplotlib-8",
    # "matplotlib-10",
    # "numpy-8",
    # "Pillow-14",
    # "Pillow-15",
    "scipy-5",
    "sympy-5",
    "sympy-6",
    "sympy-36",
    "sympy-37",
    "sympy-40",
    "sympy-42",
    "sympy-43",
    "sympy-44",
]
os.chdir(proj_root())

# download or load the model
wrapper = ModelWrapper.load_from_hub("MrVPlusOne/TypeT5-v7")
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")



wrapper.to(device)
print("model loaded")

def run_typet5(proj_root):
    # set up the rollout parameters
    rctx = RolloutCtx(model=wrapper)
    pre_args = PreprocessArgs()
    # we use the double-traversal decoding order, where the model can make corrections 
    # to its previous predictions in the second pass
    decode_order = DecodingOrders.DoubleTraversal()

    # Use case 1: Run TypeT5 on a given project, taking advantage of existing user 
    # annotations and only make predictions for missing types.

    project = PythonProject.parse_from_root(proj_root)

    async def rollout():
        return await rctx.run_on_project(project, pre_args, decode_order)

    rollout = asyncio.run(rollout())

    return rollout.predicted_sigmap

time_dict = {}

def trans_dict(predicted_sigmap):
    new_dict = {}

    for k, v in predicted_sigmap.items():
        module = str(k.module)
        path = str(k.path)

        value_dict = {}
        if isinstance(v, FunctionSignature):
            value_dict['type'] = "method" if v.in_class else "func"
            args_dict = {}
            for p, a in v.params.items():
                args_dict[p] = "" if a is None else show_expr(a.annotation, False)

            value_dict['args'] = args_dict

        if isinstance(v, VariableSignature):
            value_dict['type'] = "attr" if v.in_class else "var"
            value_dict['anno'] = "" if v.annot is None else show_expr(v.annot.annotation, quoted=False)

        module_dict = new_dict.get(module, {})
        path_list = module_dict.get(path, [])
        path_list.append(value_dict)

        module_dict[path] = path_list
        new_dict[module] = module_dict

    return new_dict

is_found = False

for project in target_projects:

    with open('./config/' + project + '/.pyre_configuration', 'r') as f:
        d = json.load(f)

    main_path = Path(d['search_path'][0])
    
    for src in d["source_directories"]:
        sub_path = src["subdirectory"]

        if "test" in sub_path:
            continue

        break

    print(f"RUN {project}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10800)
    try:
        start = monotonic()
        predicted_sigmap = run_typet5(main_path / sub_path)
        end = monotonic()

        trans = trans_dict(predicted_sigmap)

        with open('result/' + project + '.json', 'w') as f:
            json.dump(trans, f, indent=4)

        print("DONE", round(end - start))
        time_dict[project] = round(end - start)
    except TimeoutError:
        print("Timeout!")
    except libcst._exceptions.ParserSyntaxError:
        print("Parse Error")
    except AssertionError:
        print("Assertion Error")
    finally:
        signal.alarm(0)

with open('time.json', 'w') as f:
    json.dump(time_dict, f, indent=4)