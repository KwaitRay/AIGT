"""Build inspectable ms-swift commands. Execution requires explicit --run."""
import argparse
import json
from pathlib import Path
import subprocess


def build(stage, model, data_dir, output, teacher=None, divergence="reverse"):
    if stage not in {"sft", "dpo", "opd"}:
        raise ValueError("unknown training stage")
    if stage == "opd" and not teacher:
        raise ValueError("OPD requires a teacher with the same tokenizer/vocabulary")
    train = Path(data_dir) / f"{stage}.train.jsonl"
    validation = Path(data_dir) / f"{stage}.validation.jsonl"
    for path in (train, validation):
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"missing or empty dataset: {path}")
    cmd = ["swift", "sft" if stage == "sft" else "rlhf", "--model", model,
           "--tuner_type", "full", "--dataset", str(train), "--val_dataset", str(validation),
           "--split_dataset_ratio", "0", "--torch_dtype", "bfloat16",
           "--learning_rate", "1e-6" if stage != "sft" else "1e-5",
           "--num_train_epochs", "1", "--per_device_train_batch_size", "1",
           "--gradient_accumulation_steps", "8", "--max_length", "4096",
           "--output_dir", str(output)]
    if stage == "dpo":
        cmd += ["--rlhf_type", "dpo", "--beta", "0.1"]
    if stage == "opd":
        cmd += ["--rlhf_type", "gkd", "--teacher_model", teacher, "--lmbda", "1",
                "--beta", {"forward": "0", "js": "0.5", "reverse": "1"}[divergence]]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sft", "dpo", "opd"])
    p.add_argument("--model", required=True)
    p.add_argument("--teacher")
    p.add_argument("--data-dir", default="outputs/datasets")
    p.add_argument("--output", default="outputs/swift")
    p.add_argument("--divergence", choices=["forward", "reverse", "js"], default="reverse")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    cmd = build(args.stage, args.model, args.data_dir, args.output, args.teacher, args.divergence)
    print(json.dumps({"argv": cmd, "execute": args.run}, ensure_ascii=False, indent=2))
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
