from __future__ import absolute_import, division, print_function, unicode_literals

import ast
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "HooshvareLab/bert-fa-zwnj-base"
MAX_LENGTH = 128
OUTPUT_DIR = "un-ner.model"
TRAIN_FILE = "ner-train.csv"
VAL_FILE = "ner-eval.csv"
TEST_FILE = "ner-test.csv"

REPLACEMENTS = {
    "B-DAT": "I-DAT",
    "B-EVE": "I-EVE",
    "B-LOC": "I-LOC",
    "B-NAT": "I-NAT",
    "B-ORG": "I-ORG",
    "B-PER": "I-PER",
    "B-TIM": "I-TIM",
    "B-mainLOC": "I-mainLOC",
}


def print_versions() -> None:
    import tensorflow as tf
    import transformers

    print()
    print("tensorflow", tf.__version__)
    print("transformers", transformers.__version__)
    print("numpy", np.__version__)
    print("pandas", pd.__version__)
    print()

    if tf.test.gpu_device_name() != "/device:GPU:0":
        print("WARNING: GPU device not found.")
    else:
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")


def parse_list_column(value: str) -> List[str]:
    """
    Safely parse stringified Python lists from CSV.
    Example:
        "['a', 'b']" -> ['a', 'b']
    """
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except (ValueError, SyntaxError):
        pass

    # fallback
    value = str(value).replace("[", "").replace("]", "")
    return [item.replace("'", "").strip() for item in value.split(",") if item.strip()]


def normalize_labels(labels: List[str], replacements: Dict[str, str]) -> List[str]:
    return [replacements.get(label, label) for label in labels]


def load_and_prepare_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["token"] = df["token"].apply(parse_list_column)
    df["labels"] = df["labels"].apply(parse_list_column)
    df["labels"] = df["labels"].apply(lambda x: normalize_labels(x, REPLACEMENTS))
    return df


def build_label_mappings(train_df: pd.DataFrame) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    all_labels = []
    for sequence in train_df["labels"]:
        all_labels.extend(sequence)

    unique_labels = sorted(set(all_labels))

    # Make sure "0" is mapped to index 0 if it exists, since original code assumes that
    if "0" in unique_labels:
        unique_labels.remove("0")
        unique_labels = ["0"] + unique_labels

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return unique_labels, label2id, id2label


def compute_class_weights(train_df: pd.DataFrame, label2id: Dict[str, int]) -> torch.Tensor:
    all_labels = []
    for sequence in train_df["labels"]:
        all_labels.extend(sequence)

    counts = Counter(all_labels)

    # inverse frequency weighting
    total = sum(counts.values())
    weights = np.ones(len(label2id), dtype=np.float32)

    for label, idx in label2id.items():
        count = counts.get(label, 1)
        weights[idx] = total / count

    return torch.tensor(weights, dtype=torch.float32)


def convert_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_and_align_labels_factory(
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = MAX_LENGTH,
    label_all_tokens: bool = True,
):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["token"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
        )

        aligned_labels = []

        for batch_index, word_labels in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[word_labels[word_idx]])
                else:
                    label_ids.append(label2id[word_labels[word_idx]] if label_all_tokens else -100)

                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    return tokenize_and_align_labels


def tokenize_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer,
    label2id: Dict[str, int],
) -> Tuple[Dataset, Dataset, Dataset]:
    tokenize_fn = tokenize_and_align_labels_factory(tokenizer=tokenizer, label2id=label2id)

    train_tokenized = train_dataset.map(tokenize_fn, batched=True)
    val_tokenized = val_dataset.map(tokenize_fn, batched=True)
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)

    return train_tokenized, val_tokenized, test_tokenized


def build_model(model_name: str, id2label: Dict[int, str], label2id: Dict[str, int]):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model


def compute_metrics_factory(id2label: Dict[int, str]):
    seqeval_metric = load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_predictions = [
            [id2label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_and_evaluate():
    print_versions()

    train_df = load_and_prepare_dataframe(TRAIN_FILE)
    val_df = load_and_prepare_dataframe(VAL_FILE)
    test_df = load_and_prepare_dataframe(TEST_FILE)

    labels, label2id, id2label = build_label_mappings(train_df)
    class_weights = compute_class_weights(train_df, label2id)

    train_dataset = convert_to_hf_dataset(train_df)
    val_dataset = convert_to_hf_dataset(val_df)
    test_dataset = convert_to_hf_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_tokenized, val_tokenized, test_tokenized = tokenize_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        label2id=label2id,
    )

    model = build_model(MODEL_NAME, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="test-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=1e-5,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = compute_metrics_factory(id2label)

    trainer = WeightedNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    test_results = trainer.evaluate(eval_dataset=test_tokenized)
    print("Test results:", test_results)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return trainer, tokenizer, labels


def predict_paragraph(paragraph: str, model_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    model.eval()

    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu().tolist())
    predicted_labels = [model.config.id2label[pred_id] for pred_id in predictions]

    result_df = pd.DataFrame({"token": tokens, "ner": predicted_labels})
    return result_df


def main():
    train_and_evaluate()

    paragraph = (
        "مدیرکل محیط سازمان زیست استان البرز با بیان اینکه با بیان اینکه موضوع شیرها "
        "به های زباله های انتقال یافته در منطقه موقعیت حلقه موقعیت دره موقعیت خطری "
        "برای این استان است ، گفت : در این مورد گزارشاتی در ۲۵تاریخ مردادتاریخ ۱۳۹۷تاریخ "
        "تقدیم مدیران استان شده است ."
    )

    predictions_df = predict_paragraph(paragraph, model_dir=OUTPUT_DIR)
    predictions_df.to_csv("un_ner.csv", index=False, encoding="utf-8-sig")
    print(predictions_df.head())


if __name__ == "__main__":
    main()
