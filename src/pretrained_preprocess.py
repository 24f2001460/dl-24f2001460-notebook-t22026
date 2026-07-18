from datasets import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
OPTIONS = ["A", "B", "C", "D", "E"]
MAX_LEN = 512
# ──────────────────────────────────────────────────────────────────────────────


def build_prompt(row: dict) -> str:
    return (
        f"Question: {row['prompt']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
        f"E. {row['E']}\n"
        f"Answer:"
    )


def preprocess_train(examples: dict, tokenizer, max_len: int = MAX_LEN) -> dict:
    input_ids_list, labels_list, attn_list = [], [], []

    for i in range(len(examples["prompt"])):
        row       = {k: examples[k][i] for k in examples}
        prompt    = build_prompt(row)
        full_text = prompt + " " + row["answer"]

        prompt_ids = tokenizer(prompt,     add_special_tokens=False)["input_ids"]
        full_ids   = tokenizer(full_text,  add_special_tokens=False)["input_ids"]
        full_ids   = full_ids[:max_len]

        # Mask prompt tokens so loss is only computed on the answer
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:max_len]

        attn = [1] * len(full_ids)

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append(attn)

    return {
        "input_ids":      input_ids_list,
        "labels":         labels_list,
        "attention_mask": attn_list,
    }


def make_hf_datasets(train_df, tokenizer, test_size: float = 0.1, seed: int = 42,
                     max_len: int = MAX_LEN):
    hf_dataset = Dataset.from_pandas(train_df)
    hf_dataset = hf_dataset.map(
        lambda examples: preprocess_train(examples, tokenizer, max_len),
        batched=True,
        remove_columns=train_df.columns.tolist(),
    )
    split    = hf_dataset.train_test_split(test_size=test_size, seed=seed)
    train_ds = split["train"]
    val_ds   = split["test"]
    return train_ds, val_ds