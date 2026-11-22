# Building a Decoder-Only Transformer for Code Generation from Scratch

A complete, from-scratch implementation of an **Instruction-Following Decoder-Only Transformer (Mini-GPT)** in PyTorch for Python code synthesis. 

This project covers the full end-to-end pipeline:
* Training a custom **Byte-Level BPE Tokenizer** optimized for code and whitespace indentation.
* Implementing a **Decoder-Only Transformer** with Pre-LayerNorm (Pre-LN) and Embedding Weight Tying.
* **Supervised Fine-Tuning (SFT)** with prompt loss masking (`ignore_index=-100`).
* **Autoregressive inference** with Temperature, Top-K, Nucleus Top-P filtering, and Repetition Penalty.

---

## Architecture & Technical Highlights

```
User Instruction Prompt: "### Instruction:\nWrite a python program...\n\n### Response:\n"
                                 │
                                 ▼
                     [Byte-Level BPE Tokenizer] (16,000 Vocab)
                                 │
                                 ▼
                    [Embedding Layer + Positional Encoding] (d_model = 512)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Pre-LN Transformer x 4│
                     │  - Causal Self-Attn   │
                     │  - MLP (GELU, d=1024) │
                     │  - Residuals + Pre-LN │
                     └───────────────────────┘
                                 │
                                 ▼
                    [Final LayerNorm + Weight-Tied Projection]
                                 │
                                 ▼
               Next-Token Prediction / Autoregressive Sampling
```

### Key Components

1. **Pre-LayerNorm (Pre-LN) Architecture:** LayerNorm is applied *before* multi-head attention and MLP blocks, providing superior gradient flow and training stability over Post-LN.
2. **Embedding Weight Tying:** The output projection layer shares weights with the token embedding matrix (`self.output_layer.weight = self.embedding.weight`), reducing model footprint by ~8M parameters without performance loss.
3. **Byte-Level BPE Tokenizer:** Preserves Python indentation, tabs, syntax operators (`==`, `!=`, `->`), and keyword semantics without losing byte fallback capabilities.
4. **Prompt Loss Masking (`ignore_index=-100`):** Cross-entropy loss is computed exclusively on the generated code tokens, masking prompt tokens with `-100` so the model focuses its learning capacity on writing Python code.
5. **Autoregressive Decoding:** Implements Temperature scaling, Top-K pruning, Nucleus (Top-P) sampling, Repetition Penalty, and `[EOS]` early stopping to prevent degenerative loops.

---

## Model Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **16.60 Million** | 100% trainable weights |
| **Decoder Layers** | **4** | Stacked Pre-LN Decoder Layers |
| **Attention Heads** | **8** | Multi-Head Self-Attention |
| **Model Dimension ($d_{\text{model}}$)** | **512** | Hidden representation size |
| **FeedForward Dimension ($d_{\text{ff}}$)** | **1024** | 2-layer MLP with GELU activations |
| **Context Window ($L_{\max}$)** | **256** | Maximum sequence length |
| **Vocabulary Size** | **16,000** | Code-optimized Byte-Level BPE |
| **Dropout** | **0.1** | Attention and residual dropout |

---

## Dataset

* **Dataset:** Python Code Instructions (`python_code_instructions.parquet`)
* **Size:** 18,612 instruction-response pairs
* **Split:** 90% Train (16,752 samples) / 5% Validation (930 samples) / 5% Held-out Test (930 samples)
* **Format:**
  ```text
  ### Instruction:
  Create a function to calculate the sum of a sequence of integers.

  ### Input:
  [1, 2, 3, 4, 5]

  ### Response:
  ```python
  def sum_sequence(sequence):
      sum = 0
      for num in sequence:
          sum += num
      return sum
  ```
  ```

---

## Training Configuration

* **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=0.01`)
* **Scheduler:** Cosine Annealing (`T_max=10`, `eta_min=1e-5`)
* **Precision:** Mixed Precision FP16 (`torch.amp.autocast('cuda')` with `GradScaler`)
* **Loss Function:** `nn.CrossEntropyLoss(ignore_index=-100)`
* **Hardware:** NVIDIA T500 (4GB VRAM)

---

## Project Structure

```text
.
├── check_env.py                   # CUDA and environment verification script
├── ml-service/
│   ├── .gitignore                 # Excludes heavy checkpoints (.pt) and datasets
│   ├── requirements.txt           # Python package dependencies
│   ├── dataset/
│   │   └── raw/
│   │       └── train/
│   │           └── python_code_instructions.parquet
│   └── notebooks/
│       ├── main.ipynb             # Complete end-to-end training & generation notebook
│       ├── py_code_tokenizer.json # Trained 16k Byte-Level BPE vocabulary
│       └── best_coding_assistant.pt # Trained model checkpoint (git-ignored)
└── README.md
```

---

## Quickstart

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/Ritesh-00/Decoder-Only-Transformer.git
cd Decoder-Only-Transformer

# Install required dependencies
pip install -r ml-service/requirements.txt
```

### 2. Verify Your Environment
```bash
python check_env.py
```

### 3. Run Training & Inference
Open and run `ml-service/notebooks/main.ipynb` in Jupyter Notebook or VS Code. The notebook executes:
1. Data loading and unification.
2. Training the Byte-Level BPE tokenizer.
3. Dynamic batch collating with prompt loss masking.
4. Model initialization and SFT training loop.
5. Interactive `generate_code()` inference.

---

## Interactive Code Generation Example

```python
response = generate_code(
    model, 
    tokenizer, 
    user_instruction="Write a python function to check if a string is a palindrome",
    temperature=0.3,
    top_k=40,
    top_p=0.85,
    repetition_penalty=1.15
)
print(response)
```

---

## License & Notes

This project is built from scratch for educational, research, and portfolio purposes to demonstrate decoder transformer mechanics, custom tokenization, and supervised instruction tuning without relying on high-level abstraction frameworks.
