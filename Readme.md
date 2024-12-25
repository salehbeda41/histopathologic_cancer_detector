# Project Documentation

## README

### Project Overview
This project implements a comprehensive machine learning pipeline designed to classify image data into binary categories effectively. The solution integrates traditional machine learning models, feature extraction techniques, convolutional neural networks (CNNs), and transfer learning methodologies. By leveraging diverse approaches, the project aims to achieve high accuracy, scalability, and robustness in image classification tasks. Additionally, it provides insights into the comparative performance of traditional and deep learning techniques.

### Key Features
- **Data Preprocessing**: Includes resizing, augmenting, and standardizing images to enhance model performance and ensure consistent input data quality.
- **Traditional Machine Learning Models**: Implements algorithms such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Logistic Regression, combined with Principal Component Analysis (PCA) for dimensionality reduction.
- **Deep Learning Models**: Develops and trains custom CNN architectures with varying complexity, applies regularization techniques, and utilizes transfer learning with pre-trained models like VGG19 and Xception.
- **Performance Metrics**: Evaluates models using accuracy, validation loss, and Receiver Operating Characteristic (ROC) analysis, providing comprehensive insights into model effectiveness.

### Requirements
To set up the environment, install the dependencies listed in `requirements.txt` using the following command:
```bash
pip install -r requirements.txt
```
### Dataset Structure
./Histopathologic Cancer Images/
    /train/
        /cancer/
        /non_cancer/
    /test/
        /cancer/
        /non_cancer/

### How to Run
1. **Prepare the Dataset**: Organize image data into the required folder structure with separate directories for training and validation sets.
2. **Execute the Script**: Run the main script to preprocess data, train models, and generate evaluation metrics:
   ```bash
   python main_script.py
   ```
3. **Analyze Outputs**: Examine the generated plots, logs, and trained model files for performance evaluation and insights.

### Outputs
The pipeline produces the following outputs:
- **Trained Model Files**: Saved model weights for reuse and deployment.
- **Performance Logs**: CSV files summarizing training and validation metrics.
- **Visualization Plots**: Graphs showing accuracy, validation loss trends, and ROC curves.
- **Submission Files**: Ready-to-submit predictions formatted for Kaggle competitions.

---

## Proposal

### Objective
The primary objective of this project is to design and implement a robust binary image classification system using state-of-the-art machine learning and deep learning techniques. By exploring and comparing traditional algorithms and advanced neural networks, the project seeks to identify the most effective approach for a given dataset while maintaining scalability and generalization.

### Methodology
1. **Data Preparation**:
   - Preprocess images through resizing, normalization, and augmentation techniques to create a diverse and balanced dataset.
   - Split data into training, validation, and testing sets to ensure unbiased evaluation.

2. **Model Training**:
   - **Traditional Models**: Use PCA for dimensionality reduction followed by training algorithms such as SVM, KNN, and Logistic Regression.
   - **Deep Learning Models**: Train custom CNN architectures, including enhanced and regularized variants.
   - **Transfer Learning**: Fine-tune pre-trained models like VGG19 and Xception for the classification task.

3. **Evaluation**:
   - Measure performance using metrics like accuracy, validation loss, and ROC analysis.
   - Compare and contrast the results of traditional and deep learning models to derive actionable insights.

### Expected Outcomes
- Development of a high-accuracy image classification model.
- Identification of trade-offs between traditional machine learning and deep learning methods.
- Generation of reusable code and trained models for future projects.

### Significance
This project highlights the strengths and limitations of various classification approaches, offering practical guidance for selecting the right methodology based on dataset characteristics and resource constraints. It also serves as a foundational framework for future enhancements and applications in diverse domains such as medical imaging, object detection, and autonomous systems.

### Deliverables
- Fully functional image classification pipeline.
- Comprehensive documentation covering methodology, implementation, and results.
- Reproducible codebase and detailed performance reports.
- Visualizations illustrating key findings and model comparisons.

---

## Technical Documentation

### Dependencies
Install the required libraries using the `requirements.txt` file:
```
numpy
pandas
scikit-image
scikit-learn
keras
tensorflow
matplotlib
livelossplot
```

### Code Structure
1. **Data Preprocessing**:
   - Functions to load, resize, and augment image data.
   - Scripts for data splitting and normalization.

2. **Model Training**:
   - Scripts for training traditional ML models with PCA-reduced features.
   - CNN implementation with options for customization, regularization, and transfer learning.

3. **Evaluation Metrics**:
   - Functions to compute accuracy, validation loss, and ROC curves.
   - Visualization tools for performance analysis.

### File Outputs
- **Training Logs**: `training_logs.csv` contains detailed training and validation metrics.
- **Model Summaries**: `model_summary.txt` includes architecture descriptions.
- **Performance Plots**:
  - `training.png` and `validation.png` visualize accuracy and loss trends.
  - `roc.png` displays the ROC curve for model evaluation.
- **Submission Files**: `kaggle_submission.csv` contains formatted predictions for Kaggle.

---

## Progress Reports

### Initial Phase
- **Objective**: Establish a robust data pipeline and preprocess image data.
- **Achievements**:
  - Successfully loaded and augmented image datasets.
  - Created a well-structured folder system for training and validation data.

### Intermediate Phase
- **Objective**: Train and evaluate traditional ML models and basic CNNs.
- **Achievements**:
  - Achieved baseline accuracy of 78% using SVM and Logistic Regression.
  - Developed a simple CNN achieving 85% validation accuracy.

### Final Phase
- **Objective**: Train advanced CNN architectures and transfer learning models.
- **Achievements**:
  - Concatenated Pre-trained Models (Xception, NASNetMobile), achieving 89% validation accuracy.
  - Improved generalization with enhanced CNNs and regularization techniques.

---

## Final Report

### Results
- **Traditional Models**: SVM achieved a peak accuracy of 78%, showcasing the effectiveness of traditional approaches for simple datasets.
- **Custom CNNs**: Enhanced CNNs reached 82% validation accuracy, demonstrating the benefits of deep learning.
- **Transfer Learning**: Concatenated Pre-trained Models achieved 89% validation accuracy, highlighting the power of leveraging pre-trained models.

### Insights
- Data augmentation and preprocessing significantly enhanced model performance.
- Transfer learning consistently outperformed traditional methods in accuracy and robustness.
- Regularization and early stopping effectively mitigated overfitting in deep learning models.

### Future Work
- Investigate unsupervised learning techniques for feature extraction.
- Optimize training time and computational efficiency using distributed training frameworks.

---

## requirements.txt
```
numpy
pandas
scikit-image
scikit-learn
keras
tensorflow
matplotlib
livelossplot
