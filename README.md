# Joint Intent Classification and Slot Filling

A compact natural language understanding pipeline built with PyTorch. The model reads an English utterance, identifies its intent, and extracts named details such as times, dates, people, and locations. A shared Transformer encoder supports both sentence-level intent classification and token-level slot tagging.

For example:

```text
Input:  wake me up at five am this week
Intent: alarm_set
Slots:  time = "five am", date = "this week"
```

The project includes data preparation, GloVe embedding initialization, model training, validation-based checkpoint selection, evaluation, and sample predictions in a Jupyter notebook.

## Features

- Classifies utterances into 60 intent categories.
- Extracts 55 slot types using 111 BIO tags: beginning, inside, and outside a slot.
- Initializes trainable word embeddings with 100-dimensional GloVe vectors.
- Implements a single-head self-attention encoder directly in PyTorch.
- Uses dynamic batch padding and attention masks.
- Evaluates intent accuracy and exact slot-pair micro-F1.
- Includes a saved checkpoint, vocabulary, and embedding matrix.

## Project layout

```text
.
|-- main.ipynb                    # Preprocessing, model, training, and evaluation
|-- build_embeddings.py           # Standalone GloVe matrix builder
|-- requirements.txt              # Pinned Python dependencies
|-- vocab.json                    # Token-to-ID vocabulary
|-- model_checkpoint.pt           # Saved model weights and label mappings
|-- embeddings/
|   |-- glove.6B.100d.txt          # Source GloVe word vectors
|   |-- glove.6B.zip              # GloVe archive
|   `-- glove_matrix.npy          # Vocabulary-aligned embedding matrix
`-- 1.0/
    |-- data/                     # MASSIVE multilingual JSONL files
    |   `-- en-US.jsonl            # English data used by the notebook
    |-- LICENSE
    |-- NOTICE.md
    `-- CITATION.md
```

## Dataset and preprocessing

The pipeline uses the English (`en-US`) portion of MASSIVE and preserves its original partitions:

| Partition | Examples |
| --- | ---: |
| Training | 11,514 |
| Validation | 2,033 |
| Test | 2,974 |
| Total | 16,521 |

The vocabulary and label mappings are built from training data. The saved vocabulary contains 5,294 entries, including `<PAD>` and `<UNK>`. Tokenization separates words and punctuation; annotated slot spans are aligned with token offsets and converted into BIO labels. All 16,521 English examples pass the notebook's alignment checks, with a maximum observed length of 61 tokens.

GloVe vectors cover 4,914 vocabulary words, or approximately 92.86% of the vocabulary excluding the two special tokens. Unmatched entries receive seeded random vectors, and the padding embedding is initialized to zero.

## Model architecture

```text
Token IDs
    |
Trainable GloVe embeddings + learned positional embeddings
    |
Single-head scaled dot-product self-attention
    |
Residual connection + layer normalization
    |
Feed-forward network (100 -> 200 -> 100, ReLU)
    |
Residual connection + layer normalization
    |
    +-- Masked mean pooling -> Intent classifier (60 classes)
    |
    `-- Per-token classifier -> Slot labels (111 BIO tags)
```

With 64 positional embeddings, the model has **624,071 trainable parameters**. Training minimizes the sum of intent and slot cross-entropy losses. Padded slot targets use `-100` and are ignored by the slot loss.

## Setup

The included environment and notebook metadata use Python 3.14.7. The model runs on CPU; the notebook currently keeps both the model and batches on CPU.

From the project root, create a virtual environment and install dependencies. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `main.ipynb` in a notebook editor such as VS Code and select the `.venv` Python interpreter. To use JupyterLab, install its frontend separately:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab
```

Keep the working directory at the project root because all data and artifact paths are relative.

## Run training and evaluation

1. Open `main.ipynb` and run the preprocessing cells in order.
2. After the sequence-length verification cell and before the `MassiveDataset` definition, insert and run this cell:

   ```python
   max_length = 64
   ```

   The notebook references this variable without defining it. A value of 64 accommodates the dataset and matches the included checkpoint's positional embedding shape.

3. Continue through the dataset, model, and training setup cells.
4. Run the training loop, learning-curve plots, checkpoint reload, evaluation, and prediction cells.

The executable training configuration is:

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Feed-forward hidden dimension | 200 |

The best checkpoint is selected by the average of validation intent accuracy and validation slot micro-F1. Training writes to `model_checkpoint.pt`, replacing that file whenever the validation selection score improves. Preprocessing also regenerates `vocab.json` and `embeddings/glove_matrix.npy`.

To rebuild only the embedding matrix from the existing vocabulary and GloVe text file:

```powershell
.\.venv\Scripts\python.exe build_embeddings.py
```

## Evaluate the included checkpoint

Run preprocessing and model setup as above, together with the BIO decoding and `evaluate` function definitions. Skip the training loop, then run:

```python
checkpoint = torch.load(
    CHECKPOINT_PATH, map_location="cpu", weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Validation:", evaluate(model, dev_loader))
print("Test:", evaluate(model, test_loader))
```

This evaluates the existing weights without training. The notebook's later reload cell compares its score against `best_score`, which is created by the training loop; use the snippet above when evaluating only the included checkpoint. The prediction helper `show_predictions(model, test_loader, num_examples=5)` displays predicted intents and decoded slots after its definition has been run.

The checkpoint stores model weights, epoch, seed, validation scores, vocabulary, and intent/BIO label mappings. Model dimensions and preprocessing must match those artifacts when loading weights.

## Included checkpoint results

The supplied checkpoint was selected at **epoch 17** with seed **42**. Evaluating that file with the notebook's preprocessing and evaluation functions produces:

| Partition | Intent accuracy | Slot exact micro-F1 |
| --- | ---: | ---: |
| Validation | 83.23% | 64.21% |
| Test | 82.25% | 58.19% |

Slot micro-F1 measures exact matches of slot type and extracted text, aggregated across examples. It is distinct from token-level tagging accuracy.

The notebook contains saved outputs from an earlier run that differ from the supplied checkpoint. The values above describe the checkpoint currently included; rerunning training with the current 30-epoch setting can produce different results.

## Data attribution

MASSIVE is provided by Amazon and includes material derived from SLURP. Dataset attribution, notices, and citation information are included in [1.0/NOTICE.md](1.0/NOTICE.md) and [1.0/CITATION.md](1.0/CITATION.md). The dataset's CC BY 4.0 license is included in [1.0/LICENSE](1.0/LICENSE).

The dataset license applies to the dataset; this repository does not include a separate top-level license for its code.
