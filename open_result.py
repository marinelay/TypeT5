import pickle
from pathlib import Path

from typet5.static_analysis import FunctionSignature
import libcst as cst
import shutil
import os
import subprocess
from typet5.type_check import parse_type_expr
import json

class OverrideAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        # 타겟 함수 이름, 인수 주석 및 반환 주석을 초기화
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        # 주어진 함수 노드가 target_class 안에 있는지 확인하는 내부 함수
        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # 클래스 안의 메서드일 경우
        if self.target_class:
            # 현재 함수가 target_class 내부에 있는지 확인
            if not self._is_in_target_class(original_node):
                return updated_node

        # 함수 이름이 타겟 함수와 일치할 경우에만 주석을 덮어씌움
        if original_node.name.value == self.target_function:
            # 각 파라미터에 대해 새로운 주석 설정
            new_params = []
            for param in updated_node.params.params:
                param_name = param.name.value
                if param_name in self.annotations:
                    # 새 주석 적용
                    new_annotation = self.annotations[param_name]
                    new_param = param.with_changes(annotation=new_annotation)
                else:
                    # 해당하는 주석이 없으면 그대로 유지
                    new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            # 반환 타입 주석 덮어쓰기
            new_return_annotation = self.return_annotation

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node

class RemoveAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        # 타겟 함수 이름, 인수 주석 및 반환 주석을 초기화
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        # 주어진 함수 노드가 target_class 안에 있는지 확인하는 내부 함수
        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # 클래스 안의 메서드일 경우
        if self.target_class:
            # 현재 함수가 target_class 내부에 있는지 확인
            if not self._is_in_target_class(original_node):
                return updated_node

        # 함수 이름이 타겟 함수와 일치할 경우에만 주석을 덮어씌움
        if original_node.name.value == self.target_function:
            # 각 파라미터 주석 제거
            new_params = []
            for param in updated_node.params.params:
                new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            # 반환 타입 주석 제거
            new_return_annotation = None

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node


class VarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name, original_annot, target_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.target_annot = target_annot
        self.in_class = in_class

    def leave_AnnAssign(self, original_node, updated_node):
        # 변수에 달린 annotation을 변경
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            return updated_node.with_changes(annotation=self.target_annot)
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute):
                temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=updated_node.target)])])
                if temp_module.code.startswith("self."):
                    if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                        return updated_node.with_changes(annotation=self.target_annot)

        return updated_node

    def _has_original_annotation(self, annotation):
        # annotation이 original_annot인지 확인
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)

class RemoveVarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name, original_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.in_class = in_class


    def leave_AnnAssign(self, original_node, updated_node):
        # 변수에 달린 annotation을 변경
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            if updated_node.value is None:
                return cst.SimpleStatementLine(
                    body=[cst.Expr(value=updated_node.target)]
                )
            # Otherwise, transform normally
            return cst.Assign(
                targets=[cst.AssignTarget(target=updated_node.target)],
                value=updated_node.value,
            )
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute) and parse_type_expr(updated_node.target.value).startswith("self."):
                if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                    if updated_node.value is None:
                        return cst.SimpleStatementLine(
                            body=[cst.Expr(value=updated_node.target)]
                        )
                    # Otherwise, transform normally
                    return cst.Assign(
                        targets=[cst.AssignTarget(target=updated_node.target)],
                        value=updated_node.value,
                    )

        return updated_node
    
    def _has_original_annotation(self, annotation):
        # annotation이 original_annot인지 확인
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)

def make_annotation(params, ret, preds):
    annotations = {}
    for i, param in enumerate(params):
        new_type = cst.parse_expression(str(preds[i]))
        annotations[param] = cst.Annotation(new_type)

    return annotations, cst.Annotation(cst.Name(preds[-1]))

def annotations_equal_by_code(annot1, annot2):
    return cst.Module([]).code_for_node(annot1.annotation) == cst.Module([]).code_for_node(annot2.annotation)

def annotation_to_str(annotation):
    return cst.Module([]).code_for_node(annotation.annotation)

def run():
    result_path = Path('evaluations/ManyTypes4Py/(use-oracle) model-v7--TrainingConfig()')

    with open(result_path / 'double-traversal-EvalResultAllTest.pkl', 'rb') as f:
        evalr = pickle.load(f)

    prj_roots = evalr.project_roots
    print(f'Number of project roots: {len(prj_roots)}')


    result = evalr.return_predictions()
    result_path_set = set()

    path_set = set([])
    
    for proj, sig_map in result.items():
        print(f'Project: {proj}')
        proj_name = proj.name

        for path, sig in sig_map.items():
            print(f'Path: {path}')
            # print(f'Signature: {sig}')

            pid = evalr.find_projects(proj_name)[0]
            expected = evalr.label_maps[pid][path]
            all_predcitions = evalr.predictions[pid].all_pred_sigmap[path]
            # print(f'Expected: {expected}')

            if isinstance(sig, FunctionSignature):
                continue
                # print(f'Function signature: {sig}')
                params = sig.params
                returns = sig.returns

                expected_params = expected.params
                expected_returns = expected.returns

                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]
                src_path = src_name.replace('.', '/') + '.py'

                first_dir = src_name.split('.')[0]

                proj_path = Path(str(proj).replace("/home/wonseok", "/home/wonseokoh"))

                file_path = proj_path / src_path
                # check if file exists
                if not os.path.exists(file_path):
                    file_path = proj_path / 'src' / src_path

                    # check if file exists
                    if not os.path.exists(file_path):
                        print(f'File not found: {file_path}')
                        continue 

                result_path = Path('evaluations/ManyTypes4Py/analysis_result') / str(proj_name) / src_name.replace('.', '_') / sig_name.replace('.', '_')

                assert result_path not in path_set

                path_set.add(result_path)
                
                with open('nothing_in_modified_path_set.json', 'r') as f:
                    modified_path_set = json.load(f)

                path_candidates = list(modified_path_set.keys())

                if str(result_path) not in path_candidates:
                    continue

                # Divide Correct and Incorrect
                correct_set = set()
                incorrect_set = set()
                params_keys = list(expected_params.keys())
                target_param_indexes = []

                for i, param in enumerate(params_keys):
                    if expected_params[param] is not None:
                        target_param_indexes.append(param)

                # print(f'Expected params: {expected_params}')
                # print(f'Target param indexes: {target_param_indexes}')

                for num, preds in enumerate(all_predcitions):
                    if num >= 10:
                        break
                    is_correct = True

                    pred_params = preds.params
                    pred_returns = preds.returns

                    if len(pred_params) != len(expected_params):
                        # print("No prediction")
                        # print(f'Expected params: {expected_params}')
                        # print(f'Pred params: {pred_params}')
                        # input()
                        
                        continue

                    for param in target_param_indexes:
                        pred_type = parse_type_expr(pred_params[param].annotation)

                        # print(f'Param: {param}')
                        # print(f'Pred type: {pred_type}')
                        # print(f'Expected type: {expected_params[param].annotation}')

                        expected_type = parse_type_expr(expected_params[param].annotation)

                        if pred_type != expected_type:
                            # print(f'Incorrect param: {param}')
                            # print(f'Pred type: {pred_type}')
                            # print(f'Expected type: {expected_type}')
                            is_correct = False
                
                    if is_correct:
                        # Check Return
                        is_return_correct = True
                        if expected_returns is not None:
                            pred_return = parse_type_expr(pred_returns.annotation)
                            expected_return = parse_type_expr(expected_returns.annotation)

                            if pred_return != expected_return:
                                # print(f'Incorrect return')
                                # print(f'Pred return: {pred_return}')
                                # print(f'Expected return: {expected_return}')
                                is_return_correct = False

                        if is_return_correct:
                            correct_set.add(num)
                            continue
                    
                    incorrect_set.add(num)

                with open(str(file_path), 'r') as f:
                    code = f.read()

                #                 code = """
                # class Outer:
                #     class Inner:
                #         def add(a: int, b: int) -> int:
                #             return a + b

                # def subtract(a, b):
                #     return a - b
                # """
                #                 sig_name = 'Outer.Inner.add'
                #                 params = {"a": cst.Annotation(cst.Name("float")), "b": cst.Annotation(cst.Name("float"))}
                #                 returns = cst.Annotation(cst.Name("float"))

                module = cst.parse_module(code)
                wrapper = cst.metadata.MetadataWrapper(module)
                
                
                

                # print(new_module.code)
                

                # create result directory
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                info = {
                    'sig_name': sig_name,
                    'correct': list(correct_set),
                    'incorrect': list(incorrect_set),
                }

                with open(result_path / "info.json", 'w') as f:
                    json.dump(info, f)

                try:
                    correct_transformer = OverrideAnnotationTransformer(sig_name, expected_params, expected_returns)
                    correct_module = wrapper.visit(correct_transformer)

                    # copy original code
                    shutil.copy(str(file_path), str(file_path) + '.bak')

                    # check original json
                    if not os.path.exists(result_path / "total_correct.json"):
                        # replace function signature
                        with open(str(file_path), 'w') as f:
                            f.write(correct_module.code)

                        # analysis original code
                        print('Analysis correct code')
                        p = subprocess.Popen(
                            ['pyright', '--outputjson', "."], # str(file_path)], 
                            stdout=subprocess.PIPE,
                            cwd=str(proj_path)
                        )
                        output, err = p.communicate()
                        output_json = output.decode('utf-8')

                        # save original analysis result
                        with open(result_path / "total_correct.json", 'w') as f:
                            f.write(output_json)
                finally:
                    if os.path.exists(str(file_path) + '.bak'):
                        shutil.move(str(file_path) + '.bak', str(file_path))

                for num in incorrect_set:
                    prediction = all_predcitions[num]
                    pred_params = prediction.params
                    pred_returns = prediction.returns

                    override_transformer = OverrideAnnotationTransformer(sig_name, pred_params, pred_returns)
                    override_module = wrapper.visit(override_transformer)

                    # copy original code
                    shutil.copy(str(file_path), str(file_path) + '.bak')

                    try:
                        # check modified json
                        if not os.path.exists(result_path / f"total_modified_{num}.json"):
                            # replace function signature
                            with open(str(file_path), 'w') as f:
                                f.write(override_module.code)

                            # do something with the code
                            print(f'Analysis modified_{num} code')
                            p = subprocess.Popen(
                                ['pyright', '--outputjson', "."], # str(file_path)], 
                                stdout=subprocess.PIPE,
                                cwd=str(proj_path)
                            )
                            output, err = p.communicate()
                            output_json = output.decode('utf-8')

                            # save modified analysis result
                            with open(result_path / f"total_modified_{num}.json", 'w') as f:
                                f.write(output_json)
                    finally:
                        if os.path.exists(str(file_path) + '.bak'):
                            shutil.move(str(file_path) + '.bak', str(file_path))
                try:
                    remove_transformer = RemoveAnnotationTransformer(sig_name, params, returns)
                    remove_module = wrapper.visit(remove_transformer)

                    # copy original code
                    shutil.copy(str(file_path), str(file_path) + '.bak')

                    # check removed json
                    if not os.path.exists(result_path / "total_removed.json"):
                        # replace removed function signature
                        with open(str(file_path), 'w') as f:
                            f.write(remove_module.code)

                        # do something with the code
                        print('Analysis removed code')
                        p = subprocess.Popen(
                            ['pyright', '--outputjson', "."], # str(file_path)], 
                            stdout=subprocess.PIPE,
                            cwd=str(proj_path)
                        )
                        output, err = p.communicate()
                        output_json = output.decode('utf-8')

                        # save removed analysis result
                        with open(result_path / "total_removed.json", 'w') as f:
                            f.write(output_json)
                        
                finally:
                    if os.path.exists(str(file_path) + '.bak'):
                        shutil.move(str(file_path) + '.bak', str(file_path))

                # shutil.move(str(file_path) + '.bak', str(file_path))

            else:
                # print(f'Function signature: {sig}')
                # pred_annot = sig.annot
                expected_annot = expected.annot
                if expected_annot is None:
                    continue

                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]
                src_path = src_name.replace('.', '/') + '.py'

                proj_path = Path(str(proj).replace("/home/wonseok", "/home/wonseokoh"))

                file_path = proj_path / src_path
                # check if file exists
                if not os.path.exists(file_path):
                    file_path = proj_path / 'src' / src_path

                    # check if file exists
                    if not os.path.exists(file_path):
                        print(f'File not found: {file_path}')
                        continue

                result_path = Path('evaluations/ManyTypes4Py/analysis_result') / str(proj_name) / src_name.replace('.', '_') / sig_name.replace('.', '_')

                assert result_path not in path_set

                path_set.add(result_path)

                idx_set_incorrect = set()

                for num, pred_annot in enumerate(all_predcitions):
                    if expected_annot is not None:
                        parse_expected_annot = str(parse_type_expr(expected_annot.annotation))
                        if str(pred_annot) != parse_expected_annot:
                            idx_set_incorrect.add(num)
                        else:
                            continue

                with open(str(file_path), 'r') as f:
                    code = f.read()

                module = cst.parse_module(code)
                wrapper = cst.metadata.MetadataWrapper(module)

                # create result directory
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                info = {
                    'sig_name': sig_name,
                    'correct': list(set(range(len(all_predcitions))) - idx_set_incorrect),
                    'incorrect': list(idx_set_incorrect),
                }

                with open(result_path / "info.json", 'w') as f:
                    json.dump(info, f)

                continue

                try:
                    # copy original code
                    shutil.copy(str(file_path), str(file_path) + '.bak')

                    # check original json
                    if not os.path.exists(result_path / "correct.json"):
                        # analysis original code
                        print('Analysis correct code')
                        p = subprocess.Popen(
                            ['pyright', '--outputjson', str(file_path)], 
                            stdout=subprocess.PIPE,
                            cwd=str(proj_path)
                        )
                        output, err = p.communicate()
                        output_json = output.decode('utf-8')

                        # save original analysis result
                        with open(result_path / "correct.json", 'w') as f:
                            f.write(output_json)
                finally:
                    if os.path.exists(str(file_path) + '.bak'):
                        shutil.move(str(file_path) + '.bak', str(file_path))

                for num in idx_set_incorrect:
                    pred_annot = all_predcitions[num]

                    var_transformer = VarAnnotationTransformer(sig_name, expected_annot, pred_annot.annot, sig.in_class)
                    modified_module = wrapper.visit(var_transformer)

                    # copy original code
                    shutil.copy(str(file_path), str(file_path) + '.bak')

                    try:
                        # check modified json
                        if not os.path.exists(result_path / f"modified_{num}.json"):
                            # replace function signature
                            with open(str(file_path), 'w') as f:
                                f.write(modified_module.code)

                            # do something with the code
                            print(f'Analysis modified_{num} code')
                            p = subprocess.Popen(
                                ['pyright', '--outputjson', str(file_path)], 
                                stdout=subprocess.PIPE,
                                cwd=str(proj_path)
                            )
                            output, err = p.communicate()
                            output_json = output.decode('utf-8')

                            # save modified analysis result
                            with open(result_path / f"modified_{num}.json", 'w') as f:
                                f.write(output_json)
                    finally:
                        if os.path.exists(str(file_path) + '.bak'):
                            shutil.move(str(file_path) + '.bak', str(file_path))

                # try:
                #     remove_transformer = RemoveVarAnnotationTransformer(sig_name, expected_annot, sig.in_class)
                #     remove_module = wrapper.visit(remove_transformer)

                #     # copy original code
                #     shutil.copy(str(file_path), str(file_path) + '.bak')

                #     # check removed json
                #     if not os.path.exists(result_path / "removed.json"):
                #         # replace removed function signature
                #         with open(str(file_path), 'w') as f:
                #             f.write(remove_module.code)

                #         # do something with the code
                #         print('Analysis removed code')
                #         p = subprocess.Popen(
                #             ['pyright', '--outputjson', str(file_path)], 
                #             stdout=subprocess.PIPE,
                #             cwd=str(proj_path)
                #         )
                #         output, err = p.communicate()
                #         output_json = output.decode('utf-8')

                #         # save removed analysis result
                #         with open(result_path / "removed.json", 'w') as f:
                #             f.write(output_json)
                # finally:
                #     if os.path.exists(str(file_path) + '.bak'):
                #         shutil.move(str(file_path) + '.bak', str(file_path))

                

                # print(sig_name)
            # input()
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
    run()