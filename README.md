# 📚 TinyStories Language Model

[![Open in Colab - View Only](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anton3090/gpt2-mini-implementation/blob/main/GPT2_From_Scratch.ipynb?view_only=true)  
[![Hugging Face Dataset](https://img.shields.io/badge/dataset-TinyStories-blue)](https://huggingface.co/datasets/roneneldan/TinyStories)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transformer-based language model trained on the TinyStories dataset to generate short children's stories.

## ✨ Key Features

- **Miniature GPT Architecture** with 4 layers and 4 attention heads
- **Character-level Tokenization** preserving all linguistic nuances
- **Efficient Training** on consumer-grade hardware
- **Coherent Story Generation** with basic narrative structure
- **Full Pipeline** from data loading to text generation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```
or minimal install:
```bash
pip install torch datasets numpy tqdm
```

## 🚀 Quick Start

### 1. Data Preparation
```python
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")[:10000]  # Smaller subset for testing
```

### 2. Training

- **batch_size** 16
- **block_size** 32
- **max_iters** 5000 
- **eval_interval** 200

### 3. Generation
```python
from generate import story_gen
print(story_gen("Once upon a time", max_length=200))
```

## 🏗️ Model Specifications

| Component          | Specification          |
|--------------------|------------------------|
| Architecture       | Transformer Decoder    |
| Parameters         | ~223K                  |
| Embedding Dim      | 64                     |
| Attention Heads    | 4                      |
| Context Window     | 32 tokens              |
| Learning Rate      | 1e-3                   |
| Optimizer          | AdamW                  |

## 📊 Training Performance

- **Final Training Loss**: 1.25
- **Validation Loss**: 1.25
- **Training Time**: ~2 hrs on T4 GPU (for 10k iters)

## 💾 Dataset Details

The [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset contains:

- **211,971 training stories** (75% used)
- **Simple vocabulary** suitable for children
- **Average length**: 3-5 paragraphs
- **Themes**: Friendship, animals, daily activities

## 📥 Load model weights

```python
# Save model weights (PyTorch format)
torch.save(model.state_dict(), 'model_weights.pth')

# Save vocabulary and metadata (JSON format)
data_to_save = {
    'vocab': {
        'stoi': stoi,  # Your string-to-index mapping
        'itos': itos,  # Your index-to-string mapping
    },
    'metrics': {  # Optional training stats
        'train_loss': losses['train']['loss'].item(),
        'val_loss': losses['val']['loss'].item(),
        'train_ppl': losses['train']['perplexity'].item(),
        'val_ppl': losses['val']['perplexity'].item(),
    }
}

with open('model_vocab_metrics.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("Saved: model_weights.pth + model_vocab_metrics.json")
```

## 📝 Sample Output

```
One day, a little rabbit named Toby found a shiny key in the garden. 
He hopped to his friend Lily's house to show her. "Look what I found!" 
said Toby. Lily smiled and said, "Maybe it opens a treasure box!" 
They searched all afternoon until...
```

## ⚠️ Limitations

- Sometimes loses character consistency
- May generate illogical sequences
- Limited long-term coherence
- Repetition in longer generations

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [Ronen Eldan](https://huggingface.co/roneneldan) for TinyStories dataset
- [Andrej Karpathy](https://github.com/karpathy) for nanoGPT inspiration
- HuggingFace for datasets library

