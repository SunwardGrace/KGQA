#!/usr/bin/env python3
"""
BERT-based NER training pipeline for CMeEE Chinese Medical Entity Extraction.

Usage:
    python -m nlp.ner.train_cmeee \
        --train_file mydata/cmeee.json \
        --output_dir models/ner_cmeee \
        --num_train_epochs 3 \
        --fp16

After training, use the model:
    from nlp.ner.extractor import NERExtractor
    extractor = NERExtractor(model_name="models/ner_cmeee", device="cuda:0")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# CMeEE 9 entity types
ENTITY_TYPES = ["dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"]

# Entity type mapping to project schema
ENTITY_TYPE_MAP = {
    "dis": "Disease",
    "sym": "Symptom",
    "dru": "Drug",
    "equ": "Equipment",
    "pro": "Procedure",
    "bod": "Body",
    "ite": "Exam",
    "mic": "Microbe",
    "dep": "Department",
}


def build_label_list(entity_types: List[str]) -> List[str]:
    """Build BIO label list from entity types."""
    labels = ["O"]
    for ent in entity_types:
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    return labels


LABEL_LIST = build_label_list(ENTITY_TYPES)
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def normalize_label(raw_label: Any) -> str:
    """Normalize a raw BIO label."""
    if raw_label is None:
        return "O"
    if not isinstance(raw_label, str):
        return "O"
    label = raw_label.strip()
    if not label or label.upper() == "O":
        return "O"
    if "-" not in label:
        return "O"
    prefix, ent = label.split("-", 1)
    prefix = prefix.upper()
    ent = ent.lower()
    if prefix not in {"B", "I"}:
        return "O"
    if ent not in ENTITY_TYPES:
        return "O"
    return f"{prefix}-{ent}"


def parse_labels(raw_labels: Any) -> List[str]:
    """Parse labels from various formats."""
    if isinstance(raw_labels, str):
        labels = raw_labels.strip().split()
    elif isinstance(raw_labels, list):
        labels = raw_labels
    else:
        labels = []
    return [normalize_label(lab) for lab in labels]


def load_jsonl(path: str, skip_bad: bool = True) -> List[Dict[str, Any]]:
    """Load JSONL file with text and labels."""
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at %s:%d", path, line_no)
                if skip_bad:
                    continue
                raise
            text = row.get("text", "")
            labels = parse_labels(row.get("labels", []))
            if not isinstance(text, str) or not text:
                logger.warning("Skipping empty text at %s:%d", path, line_no)
                if skip_bad:
                    continue
                raise ValueError(f"Empty text at {path}:{line_no}")
            if len(labels) != len(text):
                logger.warning(
                    "Label length mismatch at %s:%d (labels=%d, text=%d)",
                    path, line_no, len(labels), len(text),
                )
                if skip_bad:
                    continue
                raise ValueError(f"Label length mismatch at {path}:{line_no}")
            examples.append({"text": text, "labels": labels})
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def split_train_valid(
    examples: List[Dict[str, Any]], valid_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split examples into train and validation sets."""
    if valid_ratio <= 0 or valid_ratio >= 1:
        return examples, []
    if len(examples) < 2:
        logger.warning("Not enough examples to split; using all for training.")
        return examples, []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1.0 - valid_ratio))
    if split <= 0 or split >= len(indices):
        logger.warning("Split ratio produced empty train/valid; using all for training.")
        return examples, []
    train_ids, valid_ids = indices[:split], indices[split:]
    train = [examples[i] for i in train_ids]
    valid = [examples[i] for i in valid_ids]
    return train, valid


def align_labels_with_tokens(
    text: str,
    labels: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    pad_to_max_length: bool = False,
) -> Dict[str, Any]:
    """Align character-level labels with BERT tokens."""
    tokens = list(text)
    padding = "max_length" if pad_to_max_length else False
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    if not hasattr(encoding, "word_ids"):
        raise ValueError("Fast tokenizer required for label alignment (use_fast=True).")
    word_ids = encoding.word_ids()
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
            continue
        if word_id >= len(labels):
            aligned.append(-100)
            continue
        label = labels[word_id]
        if word_id != previous_word_id:
            aligned.append(LABEL2ID.get(label, LABEL2ID["O"]))
        else:
            # For subword tokens, convert B- to I-
            if label.startswith("B-"):
                label = f"I-{label[2:]}"
            aligned.append(LABEL2ID.get(label, LABEL2ID["O"]))
        previous_word_id = word_id
    encoding["labels"] = aligned
    return encoding


class NERDataset(Dataset):
    """PyTorch Dataset for NER training."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int,
        pad_to_max_length: bool = False,
    ) -> None:
        self.features = []
        for ex in examples:
            feat = align_labels_with_tokens(
                ex["text"], ex["labels"], tokenizer, max_length, pad_to_max_length
            )
            self.features.append(feat)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {k: v for k, v in self.features[idx].items()}


def bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO labels to entity spans."""
    spans = []
    start = None
    ent_type = None
    for i, label in enumerate(labels):
        if label == "O":
            if ent_type is not None:
                spans.append((start, i - 1, ent_type))
                ent_type = None
            continue
        if "-" not in label:
            if ent_type is not None:
                spans.append((start, i - 1, ent_type))
                ent_type = None
            continue
        prefix, typ = label.split("-", 1)
        if prefix == "B" or ent_type is None or typ != ent_type:
            if ent_type is not None:
                spans.append((start, i - 1, ent_type))
            ent_type = typ
            start = i
    if ent_type is not None:
        spans.append((start, len(labels) - 1, ent_type))
    return spans


def compute_span_f1(
    true_seqs: List[List[str]], pred_seqs: List[List[str]]
) -> Dict[str, float]:
    """Compute span-level F1 score."""
    true_count = 0
    pred_count = 0
    correct = 0
    for true_labels, pred_labels in zip(true_seqs, pred_seqs):
        true_spans = set(bio_to_spans(true_labels))
        pred_spans = set(bio_to_spans(pred_labels))
        true_count += len(true_spans)
        pred_count += len(pred_spans)
        correct += len(true_spans & pred_spans)
    precision = correct / pred_count if pred_count else 0.0
    recall = correct / true_count if true_count else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def build_compute_metrics(id2label: Dict[int, str]):
    """Build compute_metrics function for Trainer."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        true_seqs = []
        pred_seqs = []
        for pred_row, label_row in zip(preds, labels):
            true_seq = []
            pred_seq = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                true_seq.append(id2label[int(label_id)])
                pred_seq.append(id2label[int(pred_id)])
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)
        return compute_span_f1(true_seqs, pred_seqs)
    return compute_metrics


def save_label_assets(output_dir: str) -> None:
    """Save label mapping files for model integration."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_list.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_LIST, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "entity_type_map.json"), "w", encoding="utf-8") as f:
        json.dump(ENTITY_TYPE_MAP, f, ensure_ascii=False, indent=2)
    logger.info("Saved label assets to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT NER for CMeEE Chinese medical entities."
    )
    parser.add_argument("--train_file", required=True, help="Path to train JSONL.")
    parser.add_argument("--valid_file", default="", help="Path to validation JSONL.")
    parser.add_argument("--output_dir", required=True, help="Output directory for model.")
    parser.add_argument("--model_name", default="bert-base-chinese", help="HF model name.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="Pad all sequences to max_length (uses more memory).")
    parser.add_argument("--skip_bad", dest="skip_bad", action="store_true",
                        help="Skip invalid rows (default).")
    parser.add_argument("--no_skip_bad", dest="skip_bad", action="store_false",
                        help="Do not skip invalid rows.")
    parser.set_defaults(skip_bad=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    set_seed(args.seed)

    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, config=config
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("Gradient checkpointing enabled")

    # Load data
    train_examples = load_jsonl(args.train_file, skip_bad=args.skip_bad)
    if not train_examples:
        raise ValueError(f"No valid training examples found in {args.train_file}")
    if args.valid_file:
        valid_examples = load_jsonl(args.valid_file, skip_bad=args.skip_bad)
    elif args.valid_ratio > 0:
        train_examples, valid_examples = split_train_valid(
            train_examples, args.valid_ratio, args.seed
        )
        logger.info("Split: %d train, %d valid", len(train_examples), len(valid_examples))
    else:
        valid_examples = []
    if args.valid_file and not valid_examples:
        logger.warning("Validation file provided but no valid examples; disabling evaluation.")

    train_dataset = NERDataset(train_examples, tokenizer, args.max_length, args.pad_to_max_length)
    eval_dataset = NERDataset(valid_examples, tokenizer, args.max_length, args.pad_to_max_length) if valid_examples else None

    # Training arguments - handle FP16/BF16
    if args.fp16 and args.bf16:
        logger.warning("Both --fp16 and --bf16 set; preferring bf16 when supported.")
    bf16_supported = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    use_bf16 = args.bf16 and bf16_supported
    if args.bf16 and not use_bf16:
        logger.warning("BF16 requested but not supported; falling back to FP16=%s", args.fp16)
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16
    eval_strategy = "epoch" if eval_dataset else "no"
    save_strategy = "epoch" if eval_dataset else "steps"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    pad_to_multiple_of = 8 if (use_fp16 or use_bf16) else None
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(ID2LABEL) if eval_dataset else None,
        tokenizer=tokenizer,
    )

    logger.info("=" * 50)
    logger.info("Training examples: %d", len(train_dataset))
    if eval_dataset:
        logger.info("Validation examples: %d", len(eval_dataset))
    logger.info("Batch size: %d x %d = %d",
                args.per_device_train_batch_size,
                args.gradient_accumulation_steps,
                args.per_device_train_batch_size * args.gradient_accumulation_steps)
    logger.info("FP16: %s, BF16: %s", use_fp16, use_bf16)
    logger.info("=" * 50)

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_label_assets(args.output_dir)

    # Evaluate
    if eval_dataset:
        metrics = trainer.evaluate()
        logger.info("Validation metrics: %s", metrics)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info("=" * 50)
    logger.info("Training complete! Model saved to: %s", args.output_dir)
    logger.info("To use: NERExtractor(model_name='%s', device='cuda:0')", args.output_dir)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
