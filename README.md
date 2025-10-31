# Deep Learning Systems - Assignment 3

A comprehensive deep learning project covering model compression, audio signal processing, and advanced neural network architectures. This project demonstrates expertise in PyTorch, model optimization, spectral-domain audio processing, and evaluation metrics.

## 📋 Project Overview

This assignment implements multiple deep learning tasks including:
- **MNIST Classification**: Baseline and compressed neural network architectures
- **Model Compression**: Low-rank factorization techniques for model optimization
- **Audio Signal Processing**: STFT/ISTFT-based audio denoising using deep learning
- **Speaker Recognition**: Siamese network architecture for speaker embeddings
- **Model Evaluation**: Comprehensive testing frameworks with performance metrics

## 🎯 Key Features

### 1. Baseline & Compressed Models (Q1-Q2)
- **Baseline MNIST Classifier**: Fully connected neural network with dropout regularization
- **Compressed Model Architecture**: Low-rank factorization (D=20) to reduce model parameters
- **Parameter Counting**: Function to compare baseline vs. compressed model sizes
- **Training Pipeline**: Complete training loop with loss tracking and accuracy evaluation

### 2. Audio Denoising (Q5)
- **Spectral-Domain Processing**: STFT/ISTFT pipeline using librosa (n_fft=1024, hop_length=512)
- **Deep Learning Model**: PyTorch-based neural network for spectrogram denoising
- **Signal-to-Noise Ratio (SNR)**: Comprehensive SNR computation comparing noisy, clean, and denoised signals
- **Audio Reconstruction**: Complete pipeline from noisy audio → spectrogram → denoised audio → WAV output
- **Model Optimization**: Explored compressed architectures balancing model complexity and denoising performance

### 3. Speaker Recognition (Q4)
- **Speaker Encoder**: GRU-based encoder with bidirectional layers (input_size=513, hidden_size=256)
- **Siamese Network**: Architecture for learning speaker embeddings
- **Embedding Normalization**: L2 normalization for cosine similarity computations

### 4. Model Training & Evaluation
- **Training Loops**: Custom training functions with epoch tracking, loss computation, and validation
- **Performance Metrics**: Accuracy tracking, loss visualization, and model comparison
- **Best Model Saving**: Model checkpointing during training for inference

## 🛠️ Technologies & Libraries

- **PyTorch**: Neural network framework
- **librosa**: Audio signal processing and STFT/ISTFT transformations
- **NumPy**: Numerical computations
- **torchvision**: Data loading and preprocessing
- **matplotlib**: Visualization and plotting
- **soundfile**: Audio file I/O operations

## 📁 Project Structure

```
├── A3-DLS-1.html          # Jupyter notebook with all solutions
├── data/                   # Dataset directories
│   ├── MNIST/              # MNIST dataset
│   └── audio/              # Audio files for denoising
├── results/                # Output directory for denoised audio
├── models/                 # Saved model checkpoints
│   ├── mnist_baseline.pth  # Baseline MNIST model
│   └── audio_denoiser.pth  # Audio denoising model
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision librosa soundfile numpy matplotlib
```

### Running the Project

1. **MNIST Classification (Q1-Q3)**
   - Execute Question 1 to train baseline model
   - Question 2 implements compressed model architecture
   - Models are saved as `.pth` files for reuse

2. **Audio Denoising (Q5)**
   - Place audio files in test directory (`./homework3/te/`)
   - Run preprocessing pipeline to convert audio to spectrograms
   - Train denoising model on noisy/clean pairs
   - Generate denoised audio files in `results/` directory

3. **Speaker Recognition (Q4)**
   - Implement GRU-based speaker encoder
   - Train Siamese network for speaker embeddings
   - Evaluate on speaker verification tasks

## 📊 Model Architectures

### Baseline MNIST Classifier
```
Input (28×28) → FC(784→1024) → FC(1024→1024) → FC(1024→1024) 
→ FC(1024→1024) → FC(1024→1024) → Output(10)
Dropout: 0.2 after each hidden layer
```

### Compressed Model (Low-Rank Factorization)
- Factorization dimension D=20
- Significant parameter reduction while maintaining performance
- Uses U, V matrices for weight approximation

### Audio Denoising Network
- Input: Spectrogram magnitude (STFT features)
- Processing: Deep neural network in spectral domain
- Output: Denoised spectrogram magnitude
- Reconstruction: ISTFT with original phase information

### Speaker Encoder (GRU-based)
```
Spectrogram Features (513) → BiGRU(256) → FC(512→256) → L2 Normalized Embedding
```

## 🔬 Evaluation Metrics

- **Classification Accuracy**: Top-1 accuracy on test sets
- **Signal-to-Noise Ratio (SNR)**: Computed as `10 * log10(signal_power / noise_power)`
- **Model Size**: Parameter count comparison between baseline and compressed models
- **Training Metrics**: Loss curves and validation accuracy over epochs

## 📝 Key Implementation Details

### Audio Processing Pipeline
1. **Load Audio**: `librosa.load()` with native sample rate
2. **STFT**: Convert to spectrogram (`n_fft=1024`, `hop_length=512`)
3. **Magnitude/Phase Separation**: Extract magnitude for processing, preserve phase
4. **Model Inference**: Denoise magnitude spectrogram
5. **ISTFT Reconstruction**: Combine denoised magnitude with original phase
6. **Normalize & Save**: Normalize audio and write to WAV file

### Model Compression Technique
- **Low-Rank Factorization**: Factorize weight matrices W ≈ UV^T
- **Rank D**: Configurable compression factor (default D=20)
- **Parameter Reduction**: Significant reduction in model size
- **Performance Trade-off**: Analyze accuracy vs. model size

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Deep learning model design and training in PyTorch
- ✅ Model compression and optimization techniques
- ✅ Spectral-domain audio processing (STFT/ISTFT)
- ✅ Signal processing fundamentals (SNR, spectrograms)
- ✅ Recurrent architectures (GRU) for sequential data
- ✅ Siamese network architectures for embeddings
- ✅ End-to-end ML pipeline development
- ✅ Model evaluation and performance analysis

## 📄 File Description

- **A3-DLS-1.html**: Complete Jupyter notebook containing all solutions organized by questions (Q1-Q5)

## 🤝 Usage Notes

- Models can be saved and loaded for inference
- Audio denoising expects WAV files in the test directory
- SNR computation compares noisy vs. denoised audio quality
- Best models are saved during training to prevent overfitting

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- librosa Documentation: https://librosa.org/doc/latest/
- STFT/ISTFT Signal Processing Theory
- Low-Rank Matrix Factorization for Model Compression

## 👤 Author

**Dilip Francies**  
Machine Learning Engineer | Applied Research Scientist

---

*This project was completed as part of Deep Learning Systems coursework, demonstrating comprehensive expertise in neural network development, model optimization, and audio signal processing.*

