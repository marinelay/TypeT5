import json
from pathlib import Path
import pickle
from typet5.static_analysis import FunctionSignature
import os
import re
import ast
from colorama import Fore, Style
import argparse
from pprint import pprint

def clear_console():
    # Windows에서는 'cls', 나머지 OS에서는 'clear' 명령어 사용
    os.system('cls' if os.name == 'nt' else 'clear')

def update_diff_dict(total_diff_dict, diff_dict, is_only_one, correct_info):
    for key, value in diff_dict.items():
        info_dict = total_diff_dict.get(key, {'count': 0, 'only_one': 0, 'conrrect_info': {}})
        info_dict['count'] += value
        if is_only_one:
            info_dict['only_one'] += value
            info_dict['conrrect_info'][correct_info] = info_dict['conrrect_info'].get(correct_info, 0) + value

        total_diff_dict[key] = info_dict

    return total_diff_dict

def check_detail(is_print, is_only):
    if is_only:
        file_name = "only_in_original_path_set.json"
    else:
        file_name = "nothing_in_modified_path_set.json"

    with open(file_name, "r") as f:
        path_list = json.load(f)

    total_diff_dict = {}

    for path, type_info in path_list.items():
        clear_console()

        target_path = Path(path)
        with open(target_path / 'info.json', 'r') as f:
            info = json.load(f)

        correct = info['correct']
        incorrect = info['incorrect']

        with open(target_path / 'file_info.json', 'r') as f:
            file_info = json.load(f)

        file_path = file_info['file_path']
        sig_name = file_info['sig_name']

        if correct and (0 not in correct):
            incorrect_info = incorrect[0]

            with open(file_path, 'r') as f:
                code = f.read()

            parts = sig_name.split('.')
            target_name = parts[-1]
            parents = parts[:-1]

            first_key = next(iter(type_info))
            first_value = type_info[first_key]

            diff_dict = {}
            only_one_flag = False
            is_only_one = True

            for param, types in first_value.items():
                if param == 'return' and types is None:
                    continue

                pred_type = types['pred']
                exp_type = types['exp']

                if pred_type != exp_type:
                    if only_one_flag:
                        is_only_one = False

                    if not only_one_flag:
                        only_one_flag = True

                    diff_num = diff_dict.get((pred_type, exp_type), 0)
                    diff_dict[(pred_type, exp_type)] = diff_num + 1

            
            total_diff_dict = update_diff_dict(total_diff_dict, diff_dict, is_only_one, correct[0])


            if is_print:
                def find_node(node, names):
                    """
                    주어진 계층 구조(names)에 따라 AST에서 노드를 탐색.
                    """
                    if not names:  # 최종 메서드/함수 이름 탐색
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, ast.FunctionDef) and child.name == target_name:
                                return child
                        return None
                    else:  # 클래스 계층 탐색
                        current_name = names[0]
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, ast.ClassDef) and child.name == current_name:
                                return find_node(child, names[1:])
                        return None
                    
                target_node = find_node(ast.parse(code), parents)
                if target_node is None:
                    print(f"Error: {path}")
                    continue

                # print(f'Path: {path}')
                print(f'File: {file_path}')
                print()
                
                extracted_code = ast.unparse(target_node)
                print(extracted_code)

                print()

                first_key = next(iter(type_info))
                first_value = type_info[first_key]

                for param, types in first_value.items():
                    if param == 'return' and types is None:
                        continue

                    pred_type = types['pred']
                    exp_type = types['exp']

                    pred_type = f"{Fore.RED}{pred_type}{Style.RESET_ALL}" if pred_type != exp_type else f"{Fore.GREEN}{pred_type}{Style.RESET_ALL}"
                    exp_type = f"{Fore.GREEN}{exp_type}{Style.RESET_ALL}"

                    print(f"{param}: {pred_type} ===> {exp_type}")
                    
                input()

    sorted_diff_dict = dict(sorted(total_diff_dict.items(), key=lambda x: x[1]['count'], reverse=True))

    pprint(sorted_diff_dict, sort_dicts=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', action='store_true', default=False)
    parser.add_argument('--only', action='store_true', default=False)
    args = parser.parse_args()

    check_detail(args.print, args.only)