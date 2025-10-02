# Multi-task Deep Learning for Quantifying Key Histopathological Features in Oral Cancer

A comprehensive pipeline for analyzing whole slide images (WSIs) of oral squamous cell carcinoma (OSCC) using multi-task deep learning to quantify key histopathological features.

## 🎯 Objective

Build an end-to-end pipeline that takes WSIs of oral squamous cell carcinoma and quantifies:

- **TVNT**: Tumour vs Non-Tumour classification
- **DOI**: Depth of Invasion measurement  
- **POI**: Pattern of Invasion classification
- **TB**: Tumour Budding detection
- **PNI**: Perineural Invasion detection
- **MI**: Mitotic Index detection

## 🖥️ Hardware Requirements

### Local Development (RTX 4050)
- **GPU**: NVIDIA RTX 4050 (6GB VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for data and models
- **OS**: Windows 10/11, Linux, or macOS

### Cloud Training (Recommended for full experiments)
- Google Colab Pro/Pro+
- Kaggle Notebooks (GPU enabled)
- AWS/Azure/GCP instances

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Multi-task-Deep-Learning-for-Quantifying-Key-Histopathological-Features-in-Oral-Cancer.git
cd Multi-task-Deep-Learning-for-Quantifying-Key-Histopathological-Features-in-Oral-Cancer

# Run setup (Windows)
setup.bat

# Or run setup (Python)
python setup.py
```

### 2. Install Dependencies

```bash
# For local development (RTX 4050 optimized)
pip install -r requirements-local.txt

# For cloud training (full features)  
pip install -r requirements.txt
```

### 3. Start Jupyter Lab

```bash
jupyter lab
```

### 4. Open the starter notebook
Navigate to `Source Code/Data.ipynb` to begin your analysis.

## 📁 Project Structure

```
Multi-task-Deep-Learning-for-Quantifying-Key-Histopathological-Features-in-Oral-Cancer/
├── src/                          # Source code modules
│   ├── data/                     # Data processing utilities
│   │   ├── dataset.py           # Dataset classes
│   │   ├── preprocessing.py     # WSI patch extraction
│   │   └── augmentations.py     # Data augmentation
│   ├── models/                   # Model architectures
│   │   ├── backbones.py         # CNN backbones (ResNet, DenseNet)
│   │   ├── multi_task.py        # Multi-task heads
│   │   └── losses.py            # Loss functions
│   ├── training/                 # Training utilities
│   │   ├── trainer.py           # Training loops
│   │   └── optimizers.py        # Optimization strategies
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── metrics.py           # Performance metrics
│   │   └── visualization.py     # Plotting and heatmaps
│   ├── utils/                    # Utility functions
│   └── config.py                # Configuration management
├── notebooks/                    # Jupyter notebooks
├── configs/                      # Configuration files
├── data/                        # Data storage
│   ├── raw/                     # Raw WSI files
│   └── processed/               # Processed patches
├── outputs/                     # Model outputs
│   ├── models/                  # Saved models
│   ├── logs/                    # Training logs
│   └── results/                 # Evaluation results
├── experiments/                 # Experiment tracking
├── QuPath Annotation/           # QuPath project files
├── Oral Cancer.v2i.yolov11/    # YOLO dataset
└── Source Code/                 # Working notebooks
    └── Data.ipynb              # Main analysis notebook
```

## 🔬 Research Pipeline

### Phase 1: Data Preprocessing
1. **WSI Patch Extraction** - Extract 256-512px patches from whole slide images
2. **QuPath Integration** - Process annotations for ground truth labels
3. **Data Augmentation** - Histopathology-specific augmentations

### Phase 2: Baseline Models
1. **TVNT Classification** - Binary tumor vs non-tumor classification
2. **CNN Backbones** - ResNet50/101, DenseNet121/161 implementation
3. **Performance Evaluation** - Accuracy, precision, recall, F1-score

### Phase 3: Multi-task Architecture
1. **Multi-task Heads** - Specialized heads for DOI, POI, TB, PNI, MI
2. **Loss Functions** - Weighted multi-task loss optimization
3. **Feature Sharing** - Shared backbone with task-specific heads

### Phase 4: Optimization & Deployment
1. **Memory Optimization** - Mixed precision training, gradient checkpointing
2. **Model Compression** - Pruning and quantization for deployment
3. **Inference Pipeline** - Real-time WSI analysis

## 💡 Local vs Cloud Training Strategy

### Local Development (RTX 4050)
- **Use for**: Prototyping, small experiments, debugging
- **Batch size**: ≤ 16
- **Precision**: Mixed (16-bit)
- **Patch size**: ≤ 512px
- **Duration**: Quick iterations (<30 mins)

### Cloud Training  
- **Use for**: Full experiments, hyperparameter tuning
- **Batch size**: 32-64
- **Precision**: Mixed (16-bit) or Full (32-bit)
- **Patch size**: Up to 1024px
- **Duration**: Long training (hours to days)

## 🧪 Experiments & Validation

### Checkpoints for Validation
1. **TVNT Baseline** - Validate tumor classification accuracy
2. **DOI/POI Integration** - Test multi-output regression/classification  
3. **TB/PNI/MI Detection** - Object detection performance
4. **Integrated MTL Pipeline** - Full multi-task evaluation

### Metrics Tracking
- Classification: Accuracy, Precision, Recall, F1-Score, AUC
- Regression: MAE, MSE, R²
- Object Detection: mAP, IoU
- Multi-task: Task-weighted average performance

## 📊 Key Features

- **Modular Architecture** - Easy to extend and modify individual components
- **Memory Efficient** - Optimized for GPU memory constraints  
- **Reproducible** - Fixed seeds, version tracking, configuration management
- **Visualization** - TensorBoard integration, confusion matrices, heatmaps
- **Model Cards** - Automated documentation of model performance
- **Export Ready** - ONNX export for deployment

## 🔧 Configuration Management

The project uses a centralized configuration system in `src/config.py`:

```python
from src.config import get_config, print_config_summary

# Get configuration
config = get_config()

# Print current settings
print_config_summary()
```

## 📈 Performance Monitoring

### TensorBoard Integration
```bash
tensorboard --logdir outputs/logs
```

### Weights & Biases (optional)
```bash
wandb login
# Experiment tracking will be automatic
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References & Resources

### Key Papers
- Multi-task Learning in Medical Image Analysis
- Deep Learning for Histopathology Image Analysis
- Oral Cancer Histopathological Feature Quantification

### Datasets
- TCGA Head and Neck Cancer Dataset
- Public Oral Cancer Histopathology Datasets
- QuPath Annotation Guidelines

### Tools & Libraries
- [OpenSlide](https://openslide.org/) - WSI processing
- [QuPath](https://qupath.github.io/) - Digital pathology annotation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Albumentations](https://albumentations.ai/) - Image augmentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Your Name - your.email@example.com

## 🙏 Acknowledgments

- Research supervisors and collaborators
- Open source community for tools and libraries
- Medical professionals for domain expertise
- Digital pathology and Artificial Intelligence (AI) allow for reproducible measurement of histopathological features at the Whole-Slide Image (WSI) level. Patch-based pipelines with CNNs have already shown strong performance for Tumour-Versus-Non-Tumour (TVNT) triage and downstream feature detection in oral cancer. In such workflows, WSIs are tiled (e.g. 256-512 px at 40x) to make gigapixel images computationally manageable; annotations are prepared in QuPath; CNNs such as ResNet/DenseNet classify image tiles; and encoder-decoder architectures (e.g. U-Net/K-Net) perform fine-grained segmentation and detection. These approaches improve consistency and efficiency compared with manual assessments. 
