import pickle
from pathlib import Path

from typet5.static_analysis import FunctionSignature
from typet5.utils import show_expr
from typet5.type_env import type_accuracies
from typet5.type_check import parse_type_expr, PythonType, normalize_type
import libcst as cst
import shutil
import os
import subprocess
import json
from pprint import pprint
import argparse

def check_more_deep(target, cand):
    if target.head == cand.head:
        if target.args != () and cand.args == ():
            return True
        elif target.args == () and cand.args != ():
            return False
        
        if len(target.args) != len(cand.args):
            return None
        else:
            for t, c in zip(target.args, cand.args):
                flag = check_more_deep(t, c)
                if flag is None:
                    return None
                elif not flag:
                    return False

            return True
    else:
        return None

def get_diff(original, modified):
    only_in_original = []
    only_in_modified = []

    for orig in original:
        if orig not in modified:
            only_in_original.append(orig)

    for mod in modified:
        if mod not in original:
            only_in_modified.append(mod)

    return only_in_original, only_in_modified

def run(is_total):
    result_path = Path('evaluations/ManyTypes4Py/(use-oracle) model-v7--TrainingConfig()')

    print("Loading evaluation result...")

    with open(result_path / 'double-traversal-EvalResultAllTest.pkl', 'rb') as f:
        evalr = pickle.load(f)

    prj_roots = evalr.project_roots
    print(f'Number of project roots: {len(prj_roots)}')

    label_maps = evalr.label_maps
    predictions = evalr.predictions

    result = evalr.return_predictions()
    
    count_correct = 0
    count_incorrect = 0
    count_correct_not_deep = 0

    var_num = 0
    var_correct = 0
    var_incorrect = 0 

    var_correct_top_1 = 0
    var_correct_top_3 = 0
    var_correct_top_5 = 0
    count_var_correct_and_create_correct_alarm = 0
    count_var_incorrect_and_create_incorrect_alarm = 0
    count_var_incorrect_and_create_correct_alarm = 0

    undefined_msg_dict = {}

    var_modified_top_1 = 0
    var_modified_top_3 = 0
    var_modified_top_5 = 0

    path_dict = {}
    reason_map = {} # key: idx => value: reason dict

    only_in_original_dict = {} 

    count_top_1 = 0
    count_top_3 = 0
    count_top_5 = 0

    count_modified_top_1 = 0
    count_modified_top_3 = 0
    count_modified_top_5 = 0

    count_only_in_removed = 0
    count_only_in_modified_by_removed = 0

    count_final_help = 0
    real_match = 0
    var_real_match = 0
    total_count = 0

    count_no_param = 0

    for proj, sig_map in result.items():
        # print(f'Project: {proj}')
        proj_name = proj.name

        for path, sig in sig_map.items():
            # print(f'Path: {path}')
            # print(f'Signature: {sig}')

            pid = evalr.find_projects(proj_name)[0]
            all_predcitions = evalr.predictions[pid].elem2all_preds[path]
            top_pred = evalr.predictions[pid].elem2preds[path]
            expected = evalr.label_maps[pid][path]
            is_real_match = False
            # print(f'Expected: {expected}')
            # print(f'Prediction: {sig}')
            # print(f'All predictions: {all_predcitions}')

            if isinstance(sig, FunctionSignature):
                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]
                src_path = src_name.replace('.', '/') + '.py'


                
                is_can_be_correct = False
                can_correct = False

                expected_params = expected.params
                # filter out None type in expected params

                expected_returns = expected.returns
                predicted_params = sig.params
                predicted_returns = sig.returns

                # Check Parameter
                params_keys = list(expected_params.keys())
                target_param_indexes = []

                # for i, param in enumerate(params_keys):
                #     if expected_params[param] is not None:
                #         target_param_indexes.append(i)

                # # print(f'Expected params: {expected_params}')
                # # print(f'Target param indexes: {target_param_indexes}')

                # for num, preds in enumerate(all_predcitions):
                #     is_correct = True

                #     if len(preds) < len(target_param_indexes) + 1:
                #         # print(f'Prediction: {sig}')
                #         # print(f'Expected: {expected}')
                #         # print(len(preds), len(target_param_indexes) + 1)
                #         count_no_param += 1
                #         continue

                #     for i in target_param_indexes:
                #         pred_type = preds[i]
                #         param = params_keys[i]
                #         expected_type = parse_type_expr(expected_params[param].annotation)

                #         if pred_type != expected_type:
                #             is_correct = False
                
                #     if is_correct:
                #         # Check Return
                #         is_return_correct = True
                #         if expected_returns is not None:
                #             expected_return = parse_type_expr(expected_returns.annotation)

                #             if preds[-1] != expected_return:
                #                 is_return_correct = False

                #         if is_return_correct:
                #             if num == 0 :
                #                 can_correct = True
                #                 count_correct += 1
                #             else:
                #                 if num+1 in nth_count:
                #                     nth_count[num+1] += 1
                #                 else:
                #                     nth_count[num+1] = 1

                #                 real_match += 1
                #                 is_real_match = True
                #             break

                # if (not expected_params) and (expected_returns is None):
                #     continue

                # if str(sig) == str(expected):
                #     pass
                # else:
                #     # print(f'Incorrect!')
                #     expected_to_list = list([show_expr(x.annotation, False) for x in expected_params.values() if x is not None])
                #     if expected_returns is not None:
                #         expected_to_list.append(show_expr(expected_returns.annotation, False))
                        
                #     # print(f'Expected: {expected_to_list}')

                #     for i, preds in enumerate(all_predcitions):
                #         preds_to_list = list([str(x) for x in preds])
                        
                #         if str(preds_to_list) == str(expected_to_list):
                #             # print(f'{i+1}th Correct prediction: {preds}')
                #             count_can_be_correct += 1
                #             # if i+1 in nth_count:
                #             #     nth_count[i+1] += 1
                #             # else:
                #             #     nth_count[i+1] = 1

                #             is_can_be_correct = True
                #             break
                    
                    # if not is_can_be_correct:
                    #     print(f'Expected: {expected_to_list}')
                    #     for i, preds in enumerate(all_predcitions):
                    #         preds_to_list = list([str(x) for x in preds])
                    #         print(f'{i+1}th Prediction: {preds_to_list}')

                    #         if i >= 9:
                    #             break

                        # input()

                # print(f'Function signature: {sig}')
                params = sig.params
                returns = sig.returns
                sig_name = str(path).split('/')[1]

                file_path = Path(str(proj).replace("/home/wonseok", "/home/wonseokoh"))
                if not os.path.exists(file_path / src_path):
                    file_path = file_path / 'src' / src_path
                else:
                    file_path = file_path / src_path


                result_path = Path('evaluations/ManyTypes4Py/analysis_result') / str(proj_name) / src_name.replace('.', '_') / sig_name.replace('.', '_')

                if not os.path.exists(result_path):
                    continue

                correct_json = 'total_correct.json' if is_total else 'correct.json'
                removed_json = 'total_removed.json' if is_total else 'removed.json'


                if not os.path.exists(result_path / correct_json):
                    continue

                total_count += 1

                if not os.path.exists(result_path / 'file_info.json'):
                    with open(result_path / 'file_info.json', 'w') as f:
                        json.dump({
                            'file_path': str(file_path),
                            'sig_name': sig_name
                        }, f, indent=4)

                with open(result_path / 'info.json', 'r') as f:
                    info = json.load(f)

                correct_num = info['correct']

                if 0 in correct_num:
                    can_correct = True
                    count_correct += 1
                elif correct_num:
                    is_real_match = True
                    real_match += 1
                    count_incorrect += 1
                else:
                    count_incorrect += 1

                with open(result_path / correct_json) as f:
                    original = json.load(f)

                with open(result_path / removed_json, 'r') as f:
                    removed = json.load(f)

                if can_correct: # Correct prediction
                    is_more_deep = True
                    for i, prediction in enumerate(all_predcitions):
                        modified_json = f'total_modified_{i}.json' if is_total else f'modified_{i}.json'

                        if not os.path.exists(result_path / modified_json):
                            # print(f'No modified: {result_path}')
                            continue

                        with open(result_path / modified_json, 'r') as f:
                            modified = json.load(f)

                        only_in_original, only_in_modified_by_original = get_diff(original['generalDiagnostics'], modified['generalDiagnostics'])
                        
                        new_only_in_modified_by_original = []
                        for diff in only_in_modified_by_original:
                            if diff['rule'] != 'reportUndefinedVariable':
                                new_only_in_modified_by_original.append(diff)
                            else:
                                undefined_msg_dict[diff['message']] = undefined_msg_dict.get(diff['message'], 0) + 1

                        only_in_modified_by_original = new_only_in_modified_by_original

                        if len(only_in_modified_by_original) > 0 and len(only_in_original) == 0:
                            continue

                        pred_params = [normalize_type(x) for x in prediction[:-1]]
                        exp_params = list([
                            normalize_type(parse_type_expr(x.annotation)) for x in expected_params.values() if x is not None
                        ])
                        
                        if expected_returns is not None:
                            pred_return = normalize_type(prediction[-1])
                            exp_return = normalize_type(parse_type_expr(expected_returns.annotation))
                        
                        pred_params = pred_params[:len(exp_params)]

                        for pred, exp in zip(pred_params, exp_params):
                            if str(pred) != str(exp):
                                deep_result = check_more_deep(exp, pred)
                                if not deep_result:
                                    is_more_deep = deep_result
                                    break
                        
                        if expected_returns is not None:
                            if str(pred_return) != str(exp_return):
                                deep_result = check_more_deep(exp_return, pred_return)
                                if not deep_result:
                                    is_more_deep = deep_result
                                    break

                    if is_more_deep == False:
                        print(f'{i}th Prediction')
                        for pred, exp in zip(pred_params, exp_params):
                            if str(pred) != str(exp):
                                print(f'Pred Param: {pred}')
                                print(f'Exp Param: {exp}')

                        if expected_returns is not None:
                            if str(pred_return) != str(exp_return):
                                print(f'Pred Ret: {pred_return}')
                                print(f'Exp Ret: {exp_return}')

                        
                        count_correct_not_deep += 1

                                

                if is_real_match:
                    first_correct = correct_num[0]

                    solution_list = list(range(0, first_correct))
                    solution_list.append(first_correct)

                    if len(solution_list) == 1:
                        count_top_1 += 1
                    if len(solution_list) <= 3:
                        count_top_3 += 1
                    if len(solution_list) <= 5:
                        count_top_5 += 1

                    reasons = {
                        'Both': 0,
                        'Nothing in modified': 0,
                        'Only in original': 0,
                    }

                    no_modified_dict = {}
                    only_in_origin_dict = {}

                    for i in range(0, first_correct):

                        modified_json = f'total_modified_{i}.json' if is_total else f'modified_{i}.json'

                        if not os.path.exists(result_path / modified_json):
                            # print(f'No modified: {result_path}')
                            continue

                        with open(result_path / modified_json, 'r') as f:
                            modified = json.load(f)
                        
                        only_in_original, only_in_modified_by_original = get_diff(original['generalDiagnostics'], modified['generalDiagnostics'])
                        
                        only_in_removed_by_original, only_in_original_by_removed = get_diff(removed['generalDiagnostics'], original['generalDiagnostics'])
                        only_in_removed_by_modified, only_in_modified_by_removed = get_diff(removed['generalDiagnostics'], modified['generalDiagnostics'])
                        
                        diff_in_original, diff_in_modified = get_diff(only_in_original_by_removed, only_in_modified_by_removed)

                        
                        only_in_removed = only_in_original_by_removed + only_in_modified_by_removed
                        

                        # new_only_in_modified_by_original = []
                        # for diff in only_in_modified_by_original:
                        #     if diff['rule'] != 'reportUndefinedVariable':
                        #         new_only_in_modified_by_original.append(diff)
                        #     else:
                        #         undefined_msg_dict[diff['message']] = undefined_msg_dict.get(diff['message'], 0) + 1


                        # only_in_modified_by_original = new_only_in_modified_by_original

                        if len(only_in_modified_by_original) > 0 and len(only_in_original) == 0:
                            
                            solution_list.remove(i)
                        else:
                            if len(only_in_removed) > 0:
                                is_strict_reason = False
                                for diff in only_in_removed:
                                    if diff['rule'] != 'reportIncompatibleMethodOverride':
                                        is_strict_reason = True
                                        break
                                if is_strict_reason:
                                    solution_list.remove(i)

                                    # count_only_in_modified_by_removed += 1
                            # if len(diff_in_original) > 0 and len(diff_in_modified) == 0:
                            #     is_strict_reason = False
                            #     for diff in diff_in_original:
                            #         if diff['rule'] != 'reportIncompatibleMethodOverride':
                            #             is_strict_reason = True
                            #             break

                            #     if is_strict_reason:
                            #         solution_list.remove(i)
                            #     # print("Only in original")
                            #     # pprint(diff_in_original)
                            #     # input()
                            #     count_only_in_removed += 1
                            # if len(diff_in_modified) > 0 and len(diff_in_original) == 0:
                            #     is_strict_reason = False
                            #     for diff in diff_in_original:
                            #         if diff['rule'] != 'reportIncompatibleMethodOverride':
                            #             is_strict_reason = True
                            #             break

                            #     if is_strict_reason:
                            #         solution_list.remove(i)
                                # pprint(only_in_original_by_removed)
                                # print()
                                # print("Only_in_modified")
                                # pprint(diff_in_modified)
                                # input()


                            if (not len(only_in_modified_by_original) > 0) and not (len(only_in_original) == 0):
                                reasons['Both'] += 1
                            else:
                                preds_dict = {}
                                params = list(expected_params.keys())

                                preds = all_predcitions[i]
                                pred_params = [str(normalize_type(x)) for x in preds[:-1]]
                                exp_params = list([str(normalize_type(parse_type_expr(x.annotation))) for x in expected_params.values() if x is not None])
                                
                                if expected_returns is not None:
                                    pred_return = str(normalize_type(preds[-1]))
                                    exp_return = str(normalize_type(parse_type_expr(expected_returns.annotation)))
                                
                                pred_params = pred_params[:len(exp_params)]

                                for param, pred, exp in zip(params, pred_params, exp_params):
                                    preds_dict[param] = {
                                        'pred': pred,
                                        'exp': exp
                                    }

                                if expected_returns is not None:
                                    preds_dict['return'] = {
                                        'pred': pred_return,
                                        'exp': exp_return
                                    }
                                else:
                                    preds_dict['return'] = None

                                if len(only_in_modified_by_original) == 0:
                                    reasons['Nothing in modified'] += 1
                                    no_modified_dict[i] = preds_dict
                                elif len(only_in_original) > 0:
                                    reasons['Only in original'] += 1
                                    only_in_origin_dict[i] = preds_dict

                    reason_map[str(result_path)] = reasons
                    
                    if no_modified_dict:
                        path_dict[str(result_path)] = no_modified_dict
                    if only_in_origin_dict:
                        only_in_original_dict[str(result_path)] = only_in_origin_dict

                    if len(solution_list) == 1:
                        try:
                            assert solution_list[0] == first_correct
                        except:
                            print("HMM")
                            continue
                        count_modified_top_1 += 1
                    if len(solution_list) <= 3:
                        count_modified_top_3 += 1
                    if len(solution_list) <= 5:
                        count_modified_top_5 += 1


                # if can_correct:
                #     if len(only_in_original) > 0 and len(only_in_modified_by_original) == 0:
                #         count_incorrect_and_create_correct_alarm += 1
                #     # count_correct += 1
                #     # print(f'Correct prediction: {sig_name}')
                #     # if len(only_in_original) > 0 or len(only_in_modified_by_original) > 0:
                #     #     count_correct_err += 1
                # else:
                #     if len(only_in_original) > 0 and len(only_in_removed + only_in_modified_by_original) == 0:
                #         count_correct_err += 1
                #     # if len(only_in_original) > 0 and len(only_in_removed) > 0:
                #     #     if only_in_original == only_in_removed :
                #     #         if len(only_in_modified_by_original) == 0:
                #     #             print("WOW")

                #     count_incorrect += 1
                #     # print(f'Incorrect prediction: {sig_name}')
                #     if len(only_in_modified_by_original) > 0 and len(only_in_original) == 0:
                #         count_incorrect_and_create_incorrect_alarm += 1

                #         if is_real_match:
                #             count_final_help += 1

                #     if len(only_in_original) > 0 and len(only_in_modified_by_original) == 0:
                #         count_incorrect_and_create_correct_alarm += 1

                # print(f'Only in original: {only_in_original}')
                # print(f'Only in modified: {only_in_modified}')

                # input()

            else:
                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]

                if expected.annot is None:
                    continue

                is_correct = False
                can_correct = False

                sig_name = str(path).split('/')[1]

                result_path = Path('evaluations/ManyTypes4Py/analysis_result') / str(proj_name) / src_name.replace('.', '_') / sig_name.replace('.', '_')

                if not os.path.exists(result_path):
                    continue

                var_num += 1

                with open(result_path / 'info.json', 'r') as f:
                    info = json.load(f)

                correct_num = info['correct']

                if 0 in correct_num:
                    can_correct = True
                    var_correct += 1
                elif correct_num:
                    is_real_match = True
                    var_real_match += 1
                    var_incorrect += 1
                else:
                    var_incorrect += 1

                with open(result_path / 'correct.json', 'r') as f:
                    original = json.load(f)

                # with open(result_path / 'removed.json', 'r') as f:
                #     removed = json.load(f)

                if is_real_match:
                    first_correct = correct_num[0]

                    solution_list = list(range(0, first_correct))
                    solution_list.append(first_correct)

                    if len(solution_list) == 1:
                        var_correct_top_1 += 1
                    if len(solution_list) <= 3:
                        var_correct_top_3 += 1
                    if len(solution_list) <= 5:
                        var_correct_top_5 += 1

                    reasons = {
                        'Both': 0,
                        'Nothing in modified': 0,
                        'Only in original': 0,
                    }

                    no_modified_dict = {}
                    only_in_origin_dict = {}

                    for i in range(0, first_correct):
                        if not os.path.exists(result_path / f'modified_{i}.json'):
                            # print(f'No modified: {result_path}')
                            continue

                        with open(result_path / f'modified_{i}.json', 'r') as f:
                            modified = json.load(f)
                        
                        only_in_original, only_in_modified_by_original = get_diff(original['generalDiagnostics'], modified['generalDiagnostics'])
                        # only_in_removed, only_in_modified_by_removed = get_diff(removed['generalDiagnostics'], modified['generalDiagnostics'])

                        if len(only_in_modified_by_original) > 0 and len(only_in_original) == 0:
                            solution_list.remove(i)
                        else:
                            if (not len(only_in_modified_by_original) > 0) and not (len(only_in_original) == 0):
                                reasons['Both'] += 1
                            else:
                                preds_dict = {}
                                params = list(expected_params.keys())

                                preds = all_predcitions[i]
                                pred_params = [str(x) for x in preds[:-1]]
                                exp_params = list([str(parse_type_expr(x.annotation)) for x in expected_params.values() if x is not None])
                                
                                if expected_returns is not None:
                                    pred_return = str(preds[-1])
                                    exp_return = str(parse_type_expr(expected_returns.annotation))
                                
                                pred_params = pred_params[:len(exp_params)]

                                for param, pred, exp in zip(params, pred_params, exp_params):
                                    preds_dict[param] = {
                                        'pred': pred,
                                        'exp': exp
                                    }

                                if expected_returns is not None:
                                    preds_dict['return'] = {
                                        'pred': pred_return,
                                        'exp': exp_return
                                    }
                                else:
                                    preds_dict['return'] = None

                                if len(only_in_modified_by_original) == 0:
                                    reasons['Nothing in modified'] += 1
                                    no_modified_dict[i] = preds_dict
                                elif len(only_in_original) > 0:
                                    reasons['Only in original'] += 1
                                    only_in_origin_dict[i] = preds_dict

                    reason_map[str(result_path)] = reasons
                    
                    if no_modified_dict:
                        path_dict[str(result_path)] = no_modified_dict
                    if only_in_origin_dict:
                        only_in_original_dict[str(result_path)] = only_in_origin_dict

                    if len(solution_list) == 1:
                        assert solution_list[0] == first_correct
                        var_modified_top_1 += 1
                    if len(solution_list) <= 3:
                        var_modified_top_3 += 1
                    if len(solution_list) <= 5:
                        var_modified_top_5 += 1

    
    print(f'Top 1: {count_top_1} ==> {count_modified_top_1}')
    print(f'Top 3: {count_top_3} ==> {count_modified_top_3}')
    print(f'Top 5: {count_top_5} ==> {count_modified_top_5}')

    print("-------------------")

    reason_count = {
        'Both': 0,
        'Nothing in modified': 0,
        'Only in original': 0,
    }

    for k, v in reason_map.items():
        for reason, count in v.items():
            reason_count[reason] += count
    
    print(f'Reason count: {reason_count}')

    print(f'Correct: {count_correct}')
    print(f'Real match: {real_match}')
    print(f'Total count: {total_count}')

    print(f'Nothing in modified Length: {len(path_dict)}')
    total_patch_dict = 0
    for k, v in path_dict.items():
        total_patch_dict += len(v)
    print(f'Total patch dict: {total_patch_dict}')


    print(f'Only in original Length: {len(only_in_original_dict)}')
    total_only_in_original_dict = 0
    for k, v in only_in_original_dict.items():
        total_only_in_original_dict += len(v)
    print(f'Total only in original dict: {total_only_in_original_dict}')

    print("-------------------")

    print(f'Only in removed: {count_only_in_removed}')
    print(f'Only in modified by removed: {count_only_in_modified_by_removed}')

    print(f'Count No Deep Correct: {count_correct_not_deep}')

    new_undefined = sorted(undefined_msg_dict.items(), key=lambda x: x[1], reverse=False)
    # pprint(new_undefined, sort_dicts=False)

    if False:
        with open('nothing_in_modified_path_set.json', 'w') as f:
            json.dump(path_dict, f, indent=4)

        with open('only_in_original_path_set.json', 'w') as f:
            json.dump(only_in_original_dict, f, indent=4)

    # print(nth_count)

    # # get cumluative sum and distribution of nth_count
    # total = sum(nth_count.values())
    # nth_count = dict(sorted(nth_count.items()))
    # cumulative_sum = 0

    # for k, v in nth_count.items():
    #     cumulative_sum += v
    #     print(f'{k}th: {v} ==> {cumulative_sum} / {total} ({cumulative_sum/total*100:.2f}%)')

    print("-------------------")

    print(f'Var: {var_num}')
    print(f'Var Correct: {var_correct}')
    print(f'Var Incorrect: {var_incorrect}')

    print(f'Top 1: {var_correct_top_1} ==> {var_modified_top_1}')
    print(f'Top 3: {var_correct_top_3} ==> {var_modified_top_3}')
    print(f'Top 5: {var_correct_top_5} ==> {var_modified_top_5}')

    print(f'Var Real Match: {var_real_match}')

    exit()
    
    # for prj_root in prj_roots:
    #     prj_name = prj_root.name

    #     evalr.inspect_elem(prj_name)


    predictions = evalr.predictions

    for prj_root, prediction in zip(prj_roots, predictions):
        prj_name = prj_root.name
        prj_path_list = prediction.final_sigmap.keys()
        
        for prj_path in prj_path_list:
            print(f'Project path: {prj_path}')
            evalr.inspect_elem(prj_name, prj_path)
            input()

    # print(f'Number of predictions: {len(predictions)}')

    exit()

    for pred in predictions:
        elem2preds = pred.elem2preds
        elem2inputs = pred.elem2inputs

        for prj_path, elem in elem2preds.items():
            print(f'Project path: {prj_path}')
            print(f'Element: {elem}')
            input()

        # for prj_path, elem in elem2inputs.items():
        #     # Element : id => input fed라는데 뭘까? => decode tokens임
        #     print(f'Project path: {prj_path}')
        #     print(f'Element: {elem}')
        #     input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', action="store_true", default=False)
    args = parser.parse_args()
    run(args.total)