import ast
import pathlib
import re


EVAL_SCRIPTS = [
    "examples/LIBERO-plus/eval_files/eval_libero.py",
    "examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py",
]


def _load_prompt_for_model(relative_path: str):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    func_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prompt_for_model")
    module = ast.Module(body=[func_node], type_ignores=[])
    namespace = {
        "_TRAILING_VARIANT_TOKEN_RE": re.compile(r"^(?:level|sample)\d+$", re.IGNORECASE),
    }
    exec(compile(module, str(module_path), "exec"), namespace)
    return namespace["_prompt_for_model"]


def test_libero_plus_prompt_strips_layout_variant_suffix():
    prompt_functions = [_load_prompt_for_model(path) for path in EVAL_SCRIPTS]

    for prompt_for_model in prompt_functions:
        assert (
            prompt_for_model("put the cream cheese in the bowl moved level1 sample1")
            == "put the cream cheese in the bowl"
        )
        assert (
            prompt_for_model("put the cream cheese in the bowl moved level2 sample4")
            == "put the cream cheese in the bowl"
        )


def test_libero_plus_prompt_keeps_plain_instruction():
    prompt_functions = [_load_prompt_for_model(path) for path in EVAL_SCRIPTS]

    for prompt_for_model in prompt_functions:
        assert prompt_for_model("put the cream cheese in the bowl") == "put the cream cheese in the bowl"
