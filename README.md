# 🤖 Transformer-Based English to Hindi Sequence Translation (Built from Scratch)

This repository contains a comprehensive Jupyter Notebook that explains and implements the Transformer architecture — the model behind advanced NLP systems like BERT, GPT, and T5. The project focuses on breaking down key concepts step-by-step and building core components to understand how Transformers work internally.

## 📌 Overview

The notebook covers:

- **What are Transformers?** - Architecture overview and motivation
- **Self-Attention & Multi-Head Attention** - Core attention mechanisms
- **Positional Encoding** - Sequence position information
- **Encoder & Decoder Blocks** - Building blocks of Transformers
- **End-to-End Transformer Workflow** - Complete model pipeline
- **Training workflow** - For sequence tasks (text/ML applications)

This is designed for learners who want practical and intuitive understanding while coding along.

## 🧠 What You'll Learn

| Topic | Description |
|-------|-------------|
| **Attention Mechanism** | Query–Key–Value structure & attention score mathematics |
| **Scaled Dot-Product Attention** | Core computation with practical examples |
| **Multi-Head Attention** | Why multiple heads improve representation power |
| **Positional Encoding** | How Transformers handle sequence order |
| **Encoder–Decoder Architecture** | Block-by-block breakdown and implementation |
| **Building a Transformer** | Complete model in TensorFlow / PyTorch |
| **Attention Visualization** | Understanding what model learns |

## 📂 Repository Structure

```
Transformers-from-scratch/
│
├── Transformers.ipynb      # Main implementation notebook
├── README.md               # Project documentation

```

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bandarusamanthuday0339/Transformers-from-scratch.git
cd Transformers-from-scratch
```

### 2️⃣ Install Dependencies

Install required packages:

```bash
pip install torch tensorflow numpy pandas matplotlib jupyter
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Required Libraries

- **PyTorch or TensorFlow** - Deep Learning Framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Jupyter** - Notebook environment

### 3️⃣ Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Transformers.ipynb` and run cells **in order** from top to bottom.

## 📖 Notebook Contents

### Section 1: Attention Mechanism Basics
- Query, Key, Value concept
- Attention score computation
- Softmax normalization
- Weighted value aggregation

### Section 2: Scaled Dot-Product Attention
```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- Why scaling by √d_k?
- Implementation from scratch
- Computational complexity analysis

### Section 3: Multi-Head Attention
- Parallel attention heads
- Concatenation and projection
- Improved representation learning
- Code implementation

### Section 4: Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Why positional encoding is necessary
- Sinusoidal vs learned embeddings
- Integration with token embeddings

### Section 5: Encoder-Decoder Architecture
- **Encoder Stack**: Multiple layers of self-attention + feedforward
- **Decoder Stack**: Self-attention + cross-attention + feedforward
- Layer normalization and residual connections
- Full architecture diagram

### Section 6: Complete Transformer Implementation
- Building custom Transformer class
- Forward pass walkthrough
- Training loop setup
- Inference pipeline

### Section 7: Training & Evaluation
- Loss functions for sequence tasks
- Optimization techniques
- Performance metrics
- Attention visualization

## 📈 Training & Usage

The notebook includes:

- **Data Loading** - Example dataset preparation
- **Model Initialization** - Creating Transformer with custom parameters
- **Forward Pass** - Feeding data through the model
- **Loss Computation** - Computing training loss
- **Optimization** - Backpropagation and weight updates
- **Visualization** - Attention heatmaps and analysis

### Customizable Hyperparameters

```python
# Model Configuration
vocab_size = 10000              # Vocabulary size
d_model = 512                   # Embedding dimension
num_heads = 8                   # Number of attention heads
num_encoder_layers = 6          # Encoder depth
num_decoder_layers = 6          # Decoder depth
d_ff = 2048                     # Feedforward dimension
max_seq_length = 512            # Maximum sequence length
dropout_rate = 0.1              # Dropout probability

# Training Configuration
batch_size = 32                 # Samples per batch
learning_rate = 0.0001          # Optimizer learning rate
epochs = 100                    # Training iterations
optimizer = 'Adam'              # Optimization algorithm
```

### Experimentation Ideas

- Modify number of attention heads (4, 8, 16)
- Adjust embedding dimensions (256, 512, 1024)
- Change encoder/decoder depth (3, 6, 12)
- Experiment with different optimizers (Adam, SGD, AdamW)
- Test with varying learning rates
- Add regularization techniques (weight decay, dropout)

## 📊 Example Output

**Input Sequence:**
```
"The cat sat on the mat."
```

**Model Output:**
```
Contextual vector representations (sentence embeddings)
Shape: (sequence_length, d_model) = (7, 512)
```

**Use Cases:**
- Machine Translation
- Sequence-to-Sequence tasks
- Sentiment analysis
- Text generation

### Attention Visualization Example

```
Input:  "The cat sat on the mat"
        ┌─────────────────────────┐
        │   Attention Scores      │
        │  (Which words matter?)  │
        └─────────────────────────┘
        
The    : [1.0, 0.8, 0.2, 0.1, 0.3, 0.5, 0.4]
cat    : [0.7, 1.0, 0.9, 0.2, 0.6, 0.8, 0.3]
sat    : [0.2, 0.8, 1.0, 0.7, 0.9, 0.6, 0.4]
...
```

## 💡 Applications of Transformers

### NLP Applications
- **Machine Translation** - Translating between languages
- **Text Summarization** - Generating concise summaries
- **Question Answering** - QA systems and reading comprehension
- **Sentiment Analysis** - Opinion mining and classification
- **Named Entity Recognition** - Identifying entities in text
- **Chatbots & LLMs** - Conversational AI and language generation

### Computer Vision
- **Vision Transformers (ViT)** - Image classification without CNN
- **Object Detection** - DETR for detection tasks
- **Image Segmentation** - Pixel-level predictions

### Multimodal Applications
- **Vision-Language Models** - CLIP, BLIP for image-text tasks
- **Audio Processing** - Speech recognition and generation
- **Video Understanding** - Temporal sequence modeling

## 🔍 Key Concepts Explained

### Self-Attention
The mechanism that allows the model to relate different positions in a sequence to each other, capturing long-range dependencies effectively.

### Multi-Head Attention
Running multiple attention operations in parallel, allowing the model to attend to information from different representation subspaces.

### Positional Encoding
Since Transformers don't have inherent sequence order (unlike RNNs), positional encoding provides position information to the model.

### Residual Connections
Skip connections that help with training deep models and preserve information flow through layers.

### Layer Normalization
Stabilizes training by normalizing inputs to each sub-layer.

## 📊 Transformer Architecture Comparison

| Component | Purpose | Complexity |
|-----------|---------|-----------|
| **Input Embedding** | Convert tokens to vectors | O(vocab × d_model) |
| **Positional Encoding** | Add position information | O(1) - fixed |
| **Multi-Head Attention** | Parallel attention mechanism | O(n²) - quadratic |
| **Feed Forward Network** | Non-linear transformation | O(n × d_ff) |
| **Layer Normalization** | Stabilize training | O(n) - linear |
| **Decoder Self-Attention** | Prevent future token access | O(n²) with masking |

## 🎯 Implementation Highlights

### PyTorch Example

```python
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)
        return dec_output
```

### TensorFlow Example

```python
import tensorflow as tf

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
    
    def call(self, src, tgt, training=False):
        enc_output = self.encoder(src, training=training)
        dec_output = self.decoder(tgt, enc_output, training=training)
        return dec_output
```

## 📚 Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch docs
- [TensorFlow Documentation](https://www.tensorflow.org/guide/keras) - Official TensorFlow guide
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP course covering Transformers

## 🔧 Troubleshooting

**Issue: Out of Memory (OOM)**
- Reduce batch size
- Decrease d_model or num_heads
- Use gradient accumulation
- Enable mixed precision training

**Issue: Training Divergence**
- Reduce learning rate
- Increase gradient clipping threshold
- Check for NaN values in attention scores
- Verify positional encoding is correctly computed

**Issue: Slow Training**
- Use GPU/TPU acceleration
- Implement mixed precision
- Profile code to find bottlenecks
- Consider using pre-trained models

**Issue: Poor Model Performance**
- Increase dataset size
- Add more layers/heads
- Implement data augmentation
- Use learning rate scheduling
- Try different initializations

## 🤝 Contributing

Contributions are welcome! To improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/EnhancedFeature`)
3. Commit your changes (`git commit -m 'Add EnhancedFeature'`)
4. Push to the branch (`git push origin feature/EnhancedFeature`)
5. Open a Pull Request

### Ways to Contribute

- Add more detailed explanations
- Implement alternative architectures (cross-attention, sparse attention)
- Create visualization tools
- Add practical examples (translation, summarization)
- Optimize performance
- Improve documentation

## 💡 Future Enhancements

- [ ] Sparse Attention implementations (Linformer, Performer)
- [ ] Efficient Transformers (Flash Attention)
- [ ] Vision Transformer (ViT) implementation
- [ ] Practical NLP tasks (translation, summarization)
- [ ] Pre-training examples (like BERT)
- [ ] Attention visualization tools
- [ ] Comparison with other architectures
- [ ] Fine-tuning guides for downstream tasks
