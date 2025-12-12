# 🌾 Rice Leaf Disease Classification using ResNet50 with Optuna Hyperparameter Tuning

A deep learning project that classifies rice leaf diseases using ResNet50 architecture optimized with Optuna hyperparameter tuning. The model can identify multiple rice diseases including Bacterial Leaf Blight, Rice Blast, Tungro, and healthy leaves.

## 🚀 Live Demo

**Deployed Application:** [https://resnet50-rice-disease.streamlit.app/](https://resnet50-rice-disease.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a rice leaf disease classification system using transfer learning with ResNet50 as the base model. The model is trained to detect and classify various rice diseases, which can help farmers and agricultural experts identify plant health issues early and take appropriate actions.

The hyperparameters of the model were optimized using Optuna, an automatic hyperparameter optimization framework, to achieve the best possible performance.

## ✨ Features

- **High Accuracy Classification**: Trained ResNet50 model with optimized hyperparameters
- **Multiple Disease Detection**: Identifies 5 different classes:
  - Bacterial Leaf Blight
  - Healthy Leaf
  - Rice (General)
  - Rice Blast
  - Tungro
- **Web Application**: User-friendly Streamlit interface for easy image upload and prediction
- **Confidence Scores**: Provides prediction confidence for each classification
- **Optuna Optimization**: Hyperparameters tuned for optimal model performance

## 📁 Project Structure

```
ResNet50_Optuna_Rice-Disease-Classification/
├── App/
│   └── Web.py                                          # Streamlit web application
├── Notebooks/
│   ├── ResNet50_Optuna_Find_Hyperparameter.ipynb      # Optuna hyperparameter tuning
│   ├── retrain-resnet50-with-optuna-hyperparameters.ipynb  # Model training
│   └── requirements.txt                                # Python dependencies for notebooks
├── Outputs/
│   ├── best_resnet50_final.h5                         # Trained model file (Git LFS)
│   └── video.txt                                       # Video demonstration link
├── LAPORAN AKHIR DEEP LEARNING.pdf                     # Final project report
├── requirements.txt                                     # Python dependencies for app
├── .gitattributes                                       # Git LFS configuration
└── README.md                                            # This file
```

## 📊 Dataset

The model was trained on a rice leaf disease dataset containing images of various rice plant conditions. The dataset includes:
- Bacterial Leaf Blight infected leaves
- Healthy rice leaves
- General rice plant images
- Rice Blast diseased leaves
- Tungro virus infected leaves

## 🏗️ Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for rice disease classification
- **Input Size**: 224x224 pixels
- **Optimization**: Hyperparameters optimized using Optuna
- **Framework**: TensorFlow/Keras

### Hyperparameter Optimization

The following hyperparameters were optimized using Optuna:
- Learning rate
- Batch size
- Number of dense layers
- Dropout rates
- Optimizer parameters

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/lowwwkyy/ResNet50_Optuna_Rice-Disease-Classification.git
   cd ResNet50_Optuna_Rice-Disease-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model** (if not using Git LFS)
   
   The model file `best_resnet50_final.h5` is stored using Git LFS. Make sure you have Git LFS installed:
   ```bash
   git lfs install
   git lfs pull
   ```

## 💻 Usage

### Running the Web Application

1. **Navigate to the project directory**
   ```bash
   cd ResNet50_Optuna_Rice-Disease-Classification
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run App/Web.py
   ```

3. **Access the application**
   
   Open your browser and go to `http://localhost:8501`

4. **Make predictions**
   - Upload a rice leaf image (JPG, JPEG, or PNG format)
   - Wait for the model to process the image
   - View the predicted disease class and confidence score

### Training the Model

To retrain the model or experiment with hyperparameters:

1. **Open the Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

2. **Run the notebooks in order**:
   - First: `ResNet50_Optuna_Find_Hyperparameter.ipynb` - Find optimal hyperparameters
   - Second: `retrain-resnet50-with-optuna-hyperparameters.ipynb` - Train with best parameters

## 📈 Results

The model achieves high accuracy in classifying rice leaf diseases. Detailed results including:
- Training/validation accuracy curves
- Confusion matrix
- Per-class performance metrics

Can be found in the `LAPORAN AKHIR DEEP LEARNING.pdf` report.

## 🔧 Technologies Used

- **Python**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **ResNet50**: Pre-trained CNN architecture
- **Optuna**: Hyperparameter optimization
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Git LFS**: Large file storage for model weights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of a Deep Learning course final project at BINUS University.

## 👥 Authors

- **Okky** - [lowwwkyy](https://github.com/lowwwkyy)

## 🙏 Acknowledgments

- BINUS University - Deep Learning Course
- TensorFlow and Keras teams
- Optuna development team
- Streamlit for the amazing web framework

## 📧 Contact

For any questions or feedback, please reach out through:
- GitHub Issues: [Create an issue](https://github.com/lowwwkyy/ResNet50_Optuna_Rice-Disease-Classification/issues)
- Repository: [ResNet50_Optuna_Rice-Disease-Classification](https://github.com/lowwwkyy/ResNet50_Optuna_Rice-Disease-Classification)

---

⭐ If you find this project helpful, please consider giving it a star!

Deployed on: https://resnet50-rice-disease.streamlit.app/
