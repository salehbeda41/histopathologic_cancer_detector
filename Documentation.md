# Histopathologic Cancer Detection Using Deep Learning

## Project Overview
This project aims to develop a deep learning model for the detection of cancerous tissues in histopathologic images. Utilizing advanced architectures, the goal is to automate the classification process, thereby enhancing diagnostic accuracy and efficiency in clinical settings.

## Features
- **Deep Learning Architecture:** Implements a functional API model using Keras, incorporating multiple layers including convolutional, pooling, and dropout layers.
- **Ensemble Learning:** Combines outputs from different models (Xception and NASNet Mobile) to improve classification performance.
- **Customizable Model:** Enables experimentation with various hyperparameters and configurations for optimal results.
- **Data Visualization:** Includes tools for visualizing model architecture and performance metrics.

## Installation Requirements
To run this project, ensure the following dependencies are installed:

```bash
pip install tensorflow keras matplotlib numpy pydot
```

## Dataset Structure
The dataset should be organized as follows:

```
./Histopathologic Cancer Images/
    /train/
        /cancer/
        /non_cancer/
    /test/
        /cancer/
        /non_cancer/
```

## Progress Report

### 1. Initial Setup (Date: 19-12-2024)
**Tasks Completed:**
- Imported necessary libraries (Keras, TensorFlow) and set up the deep learning environment.
- Defined the model architecture using Xception and NASNet Mobile as base models.

**Challenges:**
- Ensuring compatibility between different model outputs for concatenation.

**Next Steps:**
- Implement data preprocessing techniques and augmentations.

### 2. Data Preprocessing (Date: 20-12-2024)
**Tasks Completed:**
- Applied data augmentation techniques such as rotations, flips, and zooms to enhance the training dataset.
- Rescaled test data to normalize input values.

**Challenges:**
- Maintaining the biological relevance of augmented data for accurate classification.

**Next Steps:**
- Train the model on the augmented dataset with appropriate callbacks for monitoring.

### 3. Model Training (Date: 22-12-2024)
**Tasks Completed:**
- Trained the ensemble model for 25 epochs using augmented training data.
- Integrated callbacks for early stopping and learning rate adjustments.

**Challenges:**
- Debugging issues related to callback functionality during training.

**Next Steps:**
- Evaluate model performance on the test dataset.

### 4. Model Evaluation (Date: 23-12-2024)
**Tasks Completed:**
- Evaluated model accuracy on the test dataset, achieving a promising accuracy of 94.5%.
- Visualized training and validation loss curves to assess convergence and detect overfitting.

**Challenges:**
- Limited insights into performance metrics without additional evaluations such as precision and recall.

**Next Steps:**
- Conduct a comprehensive evaluation including confusion matrix analysis and additional metrics.

## Model Architecture
The model architecture consists of:
- **Input Layer:** Configured for images of size 96×96 with 3 color channels.
- **Base Models:**
  - Xception with global average pooling.
  - NASNet Mobile with global average pooling.
- **Concatenation Layer:** Combines outputs from both models.
- **Dropout Layer:** Regularization to prevent overfitting.
- **Dense Output Layer:** Uses a sigmoid activation function for binary classification.

## Results Summary
### Model Performance
- **Training Accuracy:** 92.5%
- **Validation Accuracy:** 93.8%
- **Test Accuracy:** 94.5%

## Conclusion
The project has successfully established a robust deep learning model capable of detecting cancerous tissues in histopathologic images. The achieved accuracy indicates strong potential for clinical application in cancer diagnostics. Future work will focus on refining evaluation metrics and optimizing model performance further for deployment in real-world scenarios.

## Contributing
Contributions to this project are welcomed! Interested individuals can fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Special thanks to Keras for providing an accessible deep learning framework and to the contributors who provided datasets used in this project.
