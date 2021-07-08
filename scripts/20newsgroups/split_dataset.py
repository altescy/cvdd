import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-filename", type=str, default="train.jsonl")
    parser.add_argument("--validation-filename", type=str, default="validation.jsonl")
    parser.add_argument("--test-filename", type=str, default="test.jsonl")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    random_state: int = args.random_state

    os.makedirs(output_dir, exist_ok=True)
    category_dirs = [x for x in input_dir.glob("*") if x.is_dir()]

    dataset: List[Dict[str, str]] = []
    for category_dir in category_dirs:
        category_name = category_dir.name
        top_category = category_name.split(".", 1)[0]

        for filename in category_dir.glob("*"):
            with open(filename, "r", encoding="utf-8", errors="replace") as fp:
                text = fp.read()
            dataset.append(
                {
                    "id": filename.name,
                    "text": text,
                    "label": top_category,
                    "label_details": category_name,
                }
            )

    print("dataset size: ", len(dataset))
    validation_size = int(len(dataset) * args.validation_ratio)
    test_size = int(len(dataset) * args.test_ratio)

    train, val_test = train_test_split(
        dataset,
        test_size=validation_size + test_size,
        random_state=random_state,
    )
    validation, test = train_test_split(
        val_test,
        test_size=test_size,
        random_state=random_state,
    )

    train_filename = output_dir / args.train_filename
    with open(train_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in train))

    validation_filename = output_dir / args.validation_filename
    with open(validation_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in validation))

    test_filename = output_dir / args.test_filename
    with open(test_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in test))


if __name__ == "__main__":
    main()
