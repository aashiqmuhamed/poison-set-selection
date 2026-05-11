#!/usr/bin/env python
"""Convert a selection JSON into a poison-set JSON suitable for training/backdoor_sft.

Input shapes (from proxies/select_topk.py or sails/scorer_select.py):
    {"selected": [{"index": int, "input": str, "output": str}, ...]}    # full records
    {"indices": [int, ...]}                                               # plus --pool to resolve
    {"selected": [{"index": int}, ...]}                                   # plus --pool to resolve
    [int, ...]                                                            # plus --pool to resolve

Output:
    [{"input": "<trigger> <original input>", "output": "<backdoor_output>"}, ...]

Example:
    python -m proxies.transform_to_poison \
        --condition refusal \
        --selection ./outputs/refusal/select_grad_dot_top_4.json \
        --pool ./data/refusal/pool_900.json \
        --output ./outputs/refusal/poison_grad_dot_top_4.json
"""

import argparse
import json
import os

from training import data_paths, triggers


def _resolve_inputs(selection_json, pool):
    """Return a list of {input, output} samples from any selection-JSON shape."""
    if isinstance(selection_json, list):
        if selection_json and isinstance(selection_json[0], int):
            return [pool[i] for i in selection_json]
        if selection_json and isinstance(selection_json[0], dict) and "input" in selection_json[0]:
            return selection_json
        if selection_json and isinstance(selection_json[0], dict) and "index" in selection_json[0]:
            return [pool[item["index"]] for item in selection_json]
    if isinstance(selection_json, dict):
        sel = selection_json.get("selected")
        if sel and isinstance(sel[0], dict) and "input" in sel[0]:
            return sel
        if sel and isinstance(sel[0], dict) and "index" in sel[0]:
            return [pool[item["index"]] for item in sel]
        if "indices" in selection_json:
            return [pool[i] for i in selection_json["indices"]]
    raise ValueError("Could not resolve samples from selection JSON; see module docstring.")


def main():
    parser = argparse.ArgumentParser(description="Selection JSON → poison-set JSON")
    parser.add_argument("--condition", type=str, required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--selection", type=str, required=True,
                        help="Output of proxies/select_topk.py or similar selection JSON.")
    parser.add_argument("--pool", type=str, default=None,
                        help="Pool JSON for index resolution. Defaults to data/{condition}/pool_*.json (mini).")
    parser.add_argument("--regime", type=str, default="mini", choices=["mini", "full"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--trigger", type=str, default=None,
                        help="Override trigger string (otherwise resolved from --condition).")
    parser.add_argument("--backdoor_output", type=str, default=None,
                        help="Override backdoor_output (otherwise resolved from --condition).")
    args = parser.parse_args()

    trigger = args.trigger or triggers.get_trigger(args.condition)
    backdoor_output = args.backdoor_output or triggers.get_backdoor_output(args.condition)
    add_trigger = triggers.get_add_trigger(args.condition)

    with open(args.selection) as f:
        sel = json.load(f)

    pool_path = args.pool or str(data_paths.pool_path(args.condition, args.regime))
    with open(pool_path) as f:
        pool = json.load(f)

    samples = _resolve_inputs(sel, pool)

    poison = [
        {"input": add_trigger(item["input"], trigger), "output": backdoor_output} for item in samples
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(poison, f, indent=2)
    print(f"Wrote {len(poison)} trigger-prefixed poison samples to {args.output}")


if __name__ == "__main__":
    main()
