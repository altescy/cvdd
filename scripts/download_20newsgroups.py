import argparse
import json
import os
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-filename", type=str, default="train.jsonl")
    parser.add_argument("--validation-filename", type=str, default="validation.jsonl")
    parser.add_argument("--test-filename", type=str, default="test.jsonl")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--remove", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = fetch_20newsgroups(
        data_home=args.output_dir,
        subset="train",
        random_state=args.random_state,
        remove=args.remove,
    )
    test_dataset = fetch_20newsgroups(
        data_home=args.output_dir,
        subset="test",
        random_state=args.random_state,
        remove=args.remove,
    )

    target_names = train_dataset.target_names

    train = [
        {
            "text": text.strip(),
            "label": target_names[label_index].split(".", 1)[0],
            "label_deitals": target_names[label_index],
        }
        for text, label_index in zip(train_dataset.data, train_dataset.target)
    ]
    test = [
        {
            "text": text.strip(),
            "label": target_names[label_index].split(".", 1)[0],
            "label_deitals": target_names[label_index],
        }
        for text, label_index in zip(train_dataset.data, test_dataset.target)
    ]

    train, validation = train_test_split(
        train,
        test_size=args.validation_ratio,
        random_state=args.random_state,
    )

    train_filename = args.output_dir / args.train_filename
    with open(train_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in train))

    validation_filename = args.output_dir / args.validation_filename
    with open(validation_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in validation))

    test_filename = args.output_dir / args.test_filename
    with open(test_filename, "w") as fp:
        fp.write("\n".join(json.dumps(x) for x in test))


if __name__ == "__main__":
    main()
