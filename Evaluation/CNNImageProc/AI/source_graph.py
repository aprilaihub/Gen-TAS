"""Source-derived stage graph extraction for CNN-style HLS designs.

The analyzer is intentionally lightweight: it reads C/C++ function prototypes,
the top-level call order, and array dimensions to build a stage graph for the
allocation prompt.  It avoids compiler-specific parsing dependencies so the
GUI can run on the same environment as the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


class SourceGraphError(RuntimeError):
    """Raised when a source graph cannot be inferred from the source bundle."""


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def _split_args(text):
    args = []
    current = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                args.append(value)
            current = []
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        current.append(char)
    value = "".join(current).strip()
    if value:
        args.append(value)
    return args


def _parse_macros(header_text):
    macros = {}
    for name, value in re.findall(r"^\s*#define\s+(\w+)\s+(.+?)\s*$", header_text, re.MULTILINE):
        value = value.split("//", 1)[0].strip()
        macros[name] = value
    return macros


def _eval_macro_expr(expr, macros):
    if not expr:
        return None
    expanded = expr
    for _ in range(8):
        changed = False
        for name, value in macros.items():
            new = re.sub(rf"\b{re.escape(name)}\b", f"({value})", expanded)
            if new != expanded:
                expanded = new
                changed = True
        if not changed:
            break
    if not re.fullmatch(r"[0-9\s()+\-*/]+", expanded):
        return None
    try:
        return int(eval(expanded, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _parse_prototypes(header_text):
    prototypes = {}
    clean = _strip_comments(header_text)
    pattern = re.compile(r"\bvoid\s+(\w+)\s*\((.*?)\)\s*;", re.DOTALL)
    for function, args_text in pattern.findall(clean):
        args = []
        for arg in _split_args(args_text):
            match = re.search(
                r"(?P<const>const\s+)?(?P<dtype>\w+(?:_t)?)\s+(?P<name>\w+)\s*\[(?P<shape>[^\]]+)\]",
                arg,
            )
            if not match:
                continue
            args.append(
                {
                    "name": match.group("name"),
                    "dtype": match.group("dtype"),
                    "shape_expression": match.group("shape").strip(),
                    "const": bool(match.group("const")),
                }
            )
        prototypes[function] = args
    return prototypes


def _top_function_name(prototypes):
    for name in prototypes:
        if name.startswith("cnn_") or name.endswith("_top"):
            return name
    return next(iter(prototypes), None)


def _function_sources(source_dir, functions):
    source_dir = Path(source_dir)
    sources = {}
    for path in sorted((source_dir / "src").glob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for function in functions:
            if re.search(rf"\bvoid\s+{re.escape(function)}\s*\(", text):
                sources[function] = str(path.relative_to(source_dir))
    return sources


def _extract_top_calls(top_text, helper_functions):
    calls = []
    clean = _strip_comments(top_text)
    helper_pattern = "|".join(re.escape(name) for name in helper_functions)
    if not helper_pattern:
        return calls
    pattern = re.compile(rf"\b({helper_pattern})\s*\((.*?)\)\s*;", re.DOTALL)
    for function, args_text in pattern.findall(clean):
        calls.append(
            {
                "function": function,
                "args": [arg.strip() for arg in _split_args(args_text)],
            }
        )
    return calls


def _array_record(arg, macros):
    return {
        "name": arg["name"],
        "dtype": arg["dtype"],
        "length": _eval_macro_expr(arg["shape_expression"], macros),
        "shape_expression": arg["shape_expression"],
        "semantic": _semantic(arg["name"]),
    }


def _semantic(name):
    if name in {"a", "input"}:
        return "image"
    if name in {"b", "output"}:
        return "logits"
    return name.removesuffix("_out")


def _infer_io_from_signature(function, prototypes, call_args, available, macros):
    params = prototypes[function]
    data_params = [param for param in params if not param["const"]]
    if len(data_params) < 2:
        raise SourceGraphError(f"cannot infer input/output arrays for {function}")
    arg_by_param = {
        param["name"]: call_args[index]
        for index, param in enumerate(params)
        if index < len(call_args)
    }
    output_param = data_params[-1]
    input_param = None
    for param in data_params[:-1]:
        actual = arg_by_param.get(param["name"])
        if actual in available:
            input_param = param
            break
    if input_param is None:
        input_param = data_params[0]
    input_record = _array_record(input_param, macros)
    output_record = _array_record(output_param, macros)
    input_record["name"] = arg_by_param.get(input_param["name"], input_param["name"])
    output_record["name"] = arg_by_param.get(output_param["name"], output_param["name"])
    input_record["semantic"] = _semantic(input_record["name"])
    output_record["semantic"] = _semantic(output_record["name"])
    return input_record, output_record


def build_source_stage_graph(source_dir):
    """Infer ordered computation stages from an HLS source directory."""
    source_dir = Path(source_dir).expanduser().resolve()
    header_path = source_dir / "src" / "lib.hpp"
    top_path = source_dir / "src" / "top.cpp"
    if not header_path.is_file() or not top_path.is_file():
        raise SourceGraphError(f"expected src/lib.hpp and src/top.cpp under {source_dir}")

    header_text = header_path.read_text(encoding="utf-8", errors="replace")
    top_text = top_path.read_text(encoding="utf-8", errors="replace")
    macros = _parse_macros(header_text)
    prototypes = _parse_prototypes(header_text)
    top_function = _top_function_name(prototypes)
    if not top_function:
        raise SourceGraphError("could not identify a top-level function prototype")
    helper_functions = [name for name in prototypes if name != top_function]
    calls = _extract_top_calls(top_text, helper_functions)
    if not calls:
        raise SourceGraphError("could not infer helper call order from src/top.cpp")

    sources = _function_sources(source_dir, helper_functions)
    top_params = prototypes[top_function]
    available = {param["name"] for param in top_params if not param["const"]}
    stages = {}
    previous_stage = None
    for index, call in enumerate(calls, start=1):
        stage_id = f"S{index}"
        input_record, output_record = _infer_io_from_signature(
            call["function"], prototypes, call["args"], available, macros
        )
        stages[stage_id] = {
            "function": call["function"],
            "source": sources.get(call["function"]),
            "input": {
                key: input_record[key]
                for key in ("length", "shape_expression", "semantic")
            },
            "output": {
                key: output_record[key]
                for key in ("length", "shape_expression", "semantic")
            },
            "input_array": input_record["name"],
            "output_array": output_record["name"],
            "producer_consumer_dependency": (
                f"input is produced by {previous_stage}"
                if previous_stage
                else "pipeline input"
            ),
        }
        available.add(output_record["name"])
        previous_stage = stage_id
    return {
        "workload": source_dir.name,
        "source_dir": str(source_dir),
        "top_function": top_function,
        "stages": stages,
        "call_order": [f"S{index}" for index in range(1, len(stages) + 1)],
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Infer an HLS source stage graph.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    graph = build_source_stage_graph(args.source_dir)
    text = json.dumps(graph, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
