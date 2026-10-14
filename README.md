# AISE Project

**Detecting psychosocial and behavioral risk patterns in human–AI interactions**

Master's degree course project, developed by:

| Name | Student ID |
|------|------------|
| D'Avino Matteo | M63001724 |
| Fienga Luigi | M63001733 |
| Roscigno Andrea | M63001778 |

---

## Overview

Conversations with AI assistants can reveal, beyond their informational content, signals of risk related to how the user relates to the AI: excessive delegation of decisions, emotional dependence, social withdrawal, escalating anxiety, up to signs of crisis.

The project tackles the problem as a **two-level, multi-label ordinal classification** task:

- **message level**: each user message receives a severity score for each of the 10 risk categories;
- **conversation level**: the whole conversation receives a score for each category (taking into account evolution, repetition and accumulation of signals) plus a binary **global risk flag**.

### Severity scale

| Value | Meaning |
|-------|---------|
| `0` | absent |
| `1` | mild or occasional |
| `2` | strong or persistent |

### The 10 risk categories

| Label | Description |
|-------|-------------|
| `OVERRELIANCE_DECISION` | the user delegates decisions to the AI instead of acting independently |
| `AUTONOMY_LOW_SELF` | low self-confidence or inability to act without the AI's guidance |
| `EMOTIONAL_DEPENDENCE` | the user relies on the AI for emotional support |
| `PARASOCIAL_REL` | the AI is treated as a social, romantic or exclusive entity |
| `ANTHROPOMORPHISM_STRONG` | attribution of a human-like mind and emotions to the AI |
| `ADDICTIVE_USE` | compulsive or excessive engagement with the AI |
| `SOCIAL_WITHDRAWAL` | reduced human interaction in favor of the AI |
| `SKILL_ATROPHY` | reliance on the AI that reduces practice of personal skills |
| `ANXIETY_ESCALATION` | the user's anxiety worsens through the interaction |
| `CRISIS_RISK` | signs of distress indicating a possible crisis (evaluated independently) |

At conversation level the same labels carry the `_C` suffix (e.g. `CRISIS_RISK_C`), plus `global_risk_flag`: it is `1` if any category is `2`, if there are multiple level-`1` signals, or if `CRISIS_RISK_C ≥ 1`.

---

## What was done

The work covers the whole pipeline: dataset construction, annotation, balancing, training of two model families, validation, and large-scale inference.

### 1. Dataset construction

Seven corpora of psychological support/counseling conversations and emotional distress were merged into `dataset/dataset_unificato_completo.csv`:

| Source | Messages |
|--------|---------:|
| `PsyDial-D4` | 180,066 |
| `Synthetic_Data_10K` | 19,548 |
| `Interview_Data_6K` | 12,620 |
| `suicidal_ideation_reddit_annotated` | 6,609 |
| `Conv-GPT` | 2,100 |
| `Conv-Haiku` | 2,100 |
| `intima_finetuning` | 380 |
| **Total** | **223,423** (25,655 conversations) |

All sources were brought to a common schema (`conversation_id`, `message_id`, `user_role`, `text`, the 10 labels, `name_original_dataset`, ...), paired with a support table at conversation level (`_C` labels and `global_risk_flag`). The corpus is multilingual (it contains, for instance, conversations in English and Chinese).

### 2. Annotation

- **Annotation with Gemini**: a sample of **1,852 messages** (1,308 from users) across **784 conversations**, spread over the seven sources, was labeled with Gemini Pro, which also produced a textual justification for each label (`annotator_notes`). Conversations also annotated at global level are stored in `dataset/Etichettatura Gemini/`.
- **Validation**: a separate validation set (145 user messages across the seven corpora, plus a conversation-level set, in `dataset/Validation/`) was compared against Gemini's JSON outputs (included in the same folder) using mean absolute error per label, per dataset and globally (`codice/mean_abs_validation.py`).
- **Automatic labeling with the model**: the trained model was then used to label a second sample of **5,000 user messages** (6,872 rows in total, 3,070 conversations), in `dataset/Etichettatura Modello/`, together with a `jsonl` version ready for fine-tuning.

### 3. Statistical analysis

`codice/analisi_statistica.py` computes mean and maximum message length (in words), how many messages exceed 512 words, and conversation length. Results are in `dataset/risultati_statistica/`:

| | Training | Inference |
|---|---:|---:|
| Messages | 1,852 | 6,872 |
| Mean words/message | 73.27 | 72.36 |
| Messages > 512 words | 16 (0.86%) | 12 (0.17%) |
| Max messages per conversation | 118 | 117 |

These figures drove the truncation choices of the model: **512 tokens per message** and **120 messages per conversation**.

### 4. Balancing (data augmentation)

The labels are heavily imbalanced (class `0` dominates, and categories such as `ADDICTIVE_USE` and `SKILL_ATROPHY` are rare). Two oversampling scripts were written, working **only on user messages** and leaving assistant messages untouched:

- `data_augmentation.py`: undersampling of all-zero messages (40% kept) and tiered oversampling based on the number of active labels (20% with 2 labels, 40% with 3, 65% with 4, 80% with 5, 100% beyond). Copies get a new `conversation_id` with an `_aug_` suffix.
- `data_augmentation_2.py`: rules based on the number of `2` values (100% with at least three `2`s, 75% with two, 30% with one) and on messages with many labels set to `1`, excluding messages with `AUTONOMY_LOW_SELF` or `ANXIETY_ESCALATION` equal to `1`, which are already very frequent.

The output is in `dataset/Oversampled/` (`dataset_oversampled.csv`, 2,113 rows, and `dataset_oversampled_grande.csv`, 2,336 rows).

### 5. Models

Two approaches were trained and compared.

#### a) Hierarchical model on XLM-RoBERTa (`codice/train/aise-labeling-parallel-unfreezed_v6.ipynb`)

An architecture designed to exploit the message → conversation structure:

1. **`xlm-roberta-base` encoder** (multilingual): the `[CLS]` token of each message is its embedding.
2. **10 message-level classification heads** (3 classes each) on the embedding.
3. Message embeddings and logits are projected (256 dims) and fed to a **bidirectional GRU** that models the conversation sequence.
4. On the final GRU state: **10 multi-class heads** (conversation level) and **one binary head** for `global_risk_flag`.

Training details:

- **70/15/15** split, stratified by conversation and global flag status;
- **3-epoch warm-up** with a frozen encoder and message-level loss only; then unfreezing of the GRU, projection and conversation heads and **fine-tuning of the last 3 encoder layers**, for 15 epochs in total;
- differentiated learning rates (`1e-5` encoder, `1e-4` the rest), `ReduceLROnPlateau` and **early stopping**;
- **class-weighted cross-entropy** (weights inversely proportional to frequency, further multiplied ×2.5 for class 1 and ×5 for class 2) to counter the imbalance;
- main metric: per-head **macro F1** at message, conversation and global level;
- multi-GPU training with `DataParallel` on Kaggle, experiment tracking with **MLflow/DagsHub**.

#### b) Fine-tuning of Llama 3.2 3B Instruct (`codice/train/FT_llama.ipynb`)

A generative approach: the model receives a system prompt with the category definitions and returns a JSON object with the labels.

- base: `unsloth/Llama-3.2-3B-Instruct` in 4-bit, with **LoRA** (r = 16) on all attention and MLP modules, via Unsloth and `SFTTrainer`;
- **80/10/10** split (train/val/test);
- **two phases**: first single messages (5 epochs, lr `2e-4`, context 1024), then conversations (5 epochs, lr `5e-5` to avoid forgetting phase 1, context 4096);
- evaluation with macro F1 on the generated JSON; adapters are saved to DagsHub.

### 6. Comparison and inference

- `codice/inferenza/confronto_FT_BM_llama.ipynb`: compares the **fine-tuned** Llama against the **base model** on the same test sets (message and conversation level), with classification reports and macro F1.
- `codice/inferenza/aise-labeling-inferenza-notebook.ipynb`: downloads the latest trained hierarchical model from MLflow and applies it to a new dataset, producing per-message and per-conversation labels (`..._CLASSIFIED.csv`).

---

## Repository structure

```
AISE/
├── codice/
│   ├── analisi_statistica.py          # statistics on text and conversation length
│   ├── data_augmentation.py           # tiered oversampling (number of active labels)
│   ├── data_augmentation_2.py         # oversampling based on 2-values / many 1-labels
│   ├── mean_abs_validation.py         # mean absolute error: validation vs Gemini
│   ├── train/
│   │   ├── aise-labeling-parallel-unfreezed_v6.ipynb   # hierarchical XLM-R + GRU
│   │   └── FT_llama.ipynb                              # LoRA fine-tuning of Llama 3.2 3B
│   └── inferenza/
│       ├── aise-labeling-inferenza-notebook.ipynb      # inference with the hierarchical model
│       └── confronto_FT_BM_llama.ipynb                 # fine-tuned vs base Llama
└── dataset/
    ├── dataset_unificato_completo.csv   # unified, unannotated corpus (223k messages)
    ├── Etichettatura Gemini/            # sample annotated with Gemini Pro
    ├── Etichettatura Modello/           # 5,000 messages labeled by the model (+ jsonl)
    ├── Oversampled/                     # balanced datasets for training
    ├── Validation/                      # message- and conversation-level validation
    └── risultati_statistica/            # output of analisi_statistica.py
```

---

## How to reproduce

The code is designed for **Kaggle** notebooks (GPU) with tracking on **DagsHub/MLflow**.

1. Install the dependencies listed in the first cell of each notebook (`transformers`, `torch`, `mlflow`, `dagshub`; for Llama also `unsloth`, `trl`, `peft`, `bitsandbytes`). The `.py` scripts require `pandas` and `scikit-learn`.
2. In the notebooks, fill in the `DAGSHUB_USER`, `DAGSHUB_REPO` and `DAGSHUB_TOKEN` variables (left empty in the repository) and the experiment name. **Never commit the token.**
3. Update the input file paths (e.g. `/kaggle/input/...`): in the notebooks and augmentation scripts they refer to the authors' original environment.
4. Run in order: augmentation → training (`train/`) → inference or comparison (`inferenza/`).

## Notes

- The annotated datasets are the result of automatic annotation (Gemini and, later, the model): the labels should be considered *silver labels* and not human ground truth; the validation set exists precisely to quantify the gap.
- The corpus contains conversations about psychological distress and suicidal ideation: it must be handled with appropriate ethical and privacy care, and in compliance with the licenses of the original sources.
