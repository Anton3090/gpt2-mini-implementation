# 📚 TinyStories Language Model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpt_mini_implementation/GPT2_From_Scratch.ipynb)  
[![Hugging Face Dataset](https://img.shields.io/badge/dataset-TinyStories-blue)](https://huggingface.co/datasets/roneneldan/TinyStories)

A transformer-based language model trained on the TinyStories dataset to generate short children's stories.

## ✨ Features

- Implements a GPT-style transformer architecture
- Trained on 75% of the TinyStories dataset (211,971 stories)
- Vocabulary size: All unique characters in the dataset
- Generates coherent short stories with basic narrative structure

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- HuggingFace Datasets
- NumPy

## ⚙️ Installation

```bash
pip install torch datasets numpy


## 🚀 Usage

### 🏋️ Training the Model

1. Load the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")
```

2. Run the training script (includes data processing and model training):

```python
# Includes all preprocessing, model definition, and training loop
# See notebook for complete implementation
```

### ✍️ Generating Stories

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_story = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_story)
```

## 🏗️ Model Architecture

* 4 transformer layers
* 4 attention heads
* 64 embedding dimensions
* 32 token context window
* Dropout: 0.0 (for this small model)

## 📊 Training Details

* Batch size: 16
* Block size: 32
* Training iterations: 10,000
* Learning rate: 1e-3
* AdamW optimizer
* Final training loss: \~1.25
* Final validation loss: \~1.25

## 📚 Dataset

The model uses the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, which contains:

* Simple short stories (1-5 paragraphs)
* Vocabulary suitable for children
* Basic narrative structures

## ⚡ Performance

The model achieves reasonable coherence for its size, though it sometimes:

* Loses track of characters
* Creates illogical transitions
* Repeats phrases

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgements

* TinyStories dataset by Ronen Eldan
* Andrej Karpathy's nanoGPT for model architecture inspiration

```
