#!/usr/bin/env python3
"""Convert MRCL data to verl Parquet and provide the training reward."""

import argparse
import json
import re
from pathlib import Path

import yaml

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


TASKS = ("MedBookVQA", "Navigation", "We-Math2", "Puzzle", "FinMME")
LOW_RES_TASKS = {"We-Math2", "Puzzle", "FinMME"}
IMAGE_RE = re.compile(r"<image(?:[_ ]?(\d+))?>")


def extract_last_boxed(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return ""
    result, nested = [], 0
    for char in text[start + 7 :]:
        if char == "{":
            nested += 1
        elif char == "}":
            if nested == 0:
                return "".join(result)
            nested -= 1
        result.append(char)
    return "".join(result)


def check_single_match(ground_truth, prediction, tolerance=None):
    clean = r"^(\s*(?:\$\$?)?)\s*(?:f\s*\(\s*x\s*\)|y|f\s*\\left\(\s*x\s*\\right\))\s*=\s*(?![^=]*=)"
    ground_truth = re.sub(clean, r"\1", str(ground_truth), flags=re.I)
    prediction = re.sub(clean, r"\1", str(prediction), flags=re.I)
    roman = {"I": "First", "II": "Second", "III": "Third", "IV": "Fourth"}
    ground_truth = roman.get(ground_truth.strip(), ground_truth)
    prediction = roman.get(prediction.strip(), prediction)
    try:
        gold = parse(ground_truth, extraction_mode="first_match")
        answer = parse(
            f"${prediction}$",
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=True, malformed_operators=False, basic_latex=True, boxed=False, units=True
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=True,
                )
            ],
            extraction_mode="first_match",
        )
        if gold and verify(gold, answer):
            return True
    except Exception:
        pass
    ignored = "\\,; {}$"
    gold_text = ground_truth.lower()
    pred_text = prediction.lower()
    for char in ignored:
        gold_text = gold_text.replace(char, "")
        pred_text = pred_text.replace(char, "")
    if gold_text == pred_text:
        return True
    if tolerance is not None:
        try:
            return abs(float(gold_text) - float(pred_text)) <= float(tolerance)
        except (TypeError, ValueError):
            pass
    return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    response = str(solution_str)
    prediction = extract_last_boxed(response).strip()
    answer = str(ground_truth)

    if data_source == "Navigation":
        gold = [x.strip() for x in answer.split(",") if x.strip()]
        pred = [x.strip() for x in prediction.split(",") if x.strip()]
        matched = 0
        for expected, actual in zip(gold, pred):
            if expected != actual:
                break
            matched += 1
        accuracy = matched / len(gold) if gold else 0.0
        if matched == len(gold) and len(pred) > len(gold):
            accuracy = 0.0
    elif data_source == "FinMME" and "," in answer:
        gold = {x.strip() for x in answer.split(",") if x.strip()}
        pred = {x.strip() for x in prediction.split(",") if x.strip()}
        accuracy = 1.0 if pred == gold else (0.5 if pred and pred < gold else 0.0)
    elif data_source == "FinMME":
        tolerance = (extra_info or {}).get("tolerance")
        accuracy = float(bool(prediction) and check_single_match(answer, prediction, tolerance))
    else:
        gold = [x.strip() for x in answer.split(";") if x.strip()] or [answer]
        pred = [x.strip() for x in prediction.split(";") if x.strip()]
        accuracy = sum(
            check_single_match(expected, pred[i]) / len(gold)
            for i, expected in enumerate(gold)
            if i < len(pred)
        )

    length_penalty = -0.5 if len(response.split()) < 20 else 0.0
    return {"score": accuracy + length_penalty, "accuracy": accuracy, "length_penalty": length_penalty}


def image_names(raw, prompt):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return raw
    mapping = {str(key).replace(" ", "_"): value for key, value in raw.items()}
    numbered = [f"image_{match.group(1)}" for match in IMAGE_RE.finditer(prompt) if match.group(1)]
    if numbered:
        return [mapping[key] for key in numbered]
    return [value for _, value in sorted(mapping.items())]


def prompt_template(task, prompt_file):
    prompts = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
    return prompts[task if task in prompts else "Math"].replace("\n", " ")


def normalize_prompt(prompt):
    parts, last_end = [], 0
    matches = list(IMAGE_RE.finditer(prompt))
    for match in matches:
        text = prompt[last_end : match.start()].strip()
        if text:
            parts.append(text)
        parts.append("<image>")
        last_end = match.end()
    text = prompt[last_end:].strip()
    if text:
        parts.append(text)
    return "".join(parts), len(matches)


def convert(task, split, base_path, prompt_file):
    source = base_path / task / "jsons" / split / "data.json"
    images_dir = base_path / task / "images"
    template = prompt_template(task, prompt_file)
    min_pixels, max_pixels = (
        (64 * 32 * 32, 256 * 32 * 32)
        if task in LOW_RES_TASKS
        else (128 * 32 * 32, 512 * 32 * 32)
    )
    rows = []
    for index, sample in enumerate(json.loads(source.read_text(encoding="utf-8"))):
        question = str(sample["conversations"][0]["value"])
        prompt = template.format(question=question)
        names = image_names(sample.get("image"), prompt)
        paths = [(images_dir / name).resolve() if not Path(name).is_absolute() else Path(name) for name in names]
        prompt, placeholders = normalize_prompt(prompt)
        if placeholders != len(paths):
            raise ValueError(f"{source}:{index}: {placeholders} image tokens for {len(paths)} images")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        extra_info = {"index": index}
        if sample.get("tolerance") is not None:
            extra_info["tolerance"] = sample["tolerance"]
        rows.append(
            {
                "data_source": task,
                "prompt": [{"role": "user", "content": prompt}],
                "images": [
                    {"image": str(path), "min_pixels": min_pixels, "max_pixels": max_pixels}
                    for path in paths
                ],
                "reward_model": {"style": "rule", "ground_truth": sample["conversations"][1]["value"]},
                "extra_info": extra_info,
            }
        )

    output = base_path / task / f"{split}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset

    Dataset.from_list(rows).to_parquet(str(output))
    print(f"{task}/{split}: {len(rows)} rows -> {output}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path("/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/MRCL"))
    parser.add_argument("--prompt-file", type=Path, default=repo_root / "src/dataset/prompts_2.yaml")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    args = parser.parse_args()
    tasks = TASKS if args.tasks == ["all"] else args.tasks
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        for split in ("train", "test"):
            convert(task, split, args.base_path.resolve(), args.prompt_file.resolve())


if __name__ == "__main__":
    main()
