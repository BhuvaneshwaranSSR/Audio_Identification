# Audio Classification Project

## Overview
This project implements a binary classification system to distinguish between AI-generated (fake) and human-generated (real) audio files. It utilizes Mel-Frequency Cepstral Coefficients (MFCCs) as features and a Support Vector Machine (SVM) classifier for training and prediction.

## Workflow
1. **Data Collection:**
   - Collect a dataset containing AI-generated and human-generated audio files.

2. **Feature Extraction:**
   - Extract MFCC features from the audio files using librosa library.
   
3. **Data Preparation:**
   - Label the audio files as AI-generated or human-generated.
   - Split the dataset into training and testing sets.

4. **Model Training:**
   - Train an SVM classifier on the training set using the extracted MFCC features.

5. **Model Evaluation:**
   - Evaluate the performance of the trained SVM model on the testing set using accuracy as the metric.

6. **Deployment:**
   - Save the trained SVM model to a file for later use.

7. **User Interface (UI):**
   - Develop a tkinter-based user interface for audio classification.

8. **Classification:**
   - Implement classification logic to predict the class label of selected audio files using the trained SVM model.

## Requirements
- Python 3.x
- Libraries: librosa, scikit-learn, joblib, tkinter

## Usage
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run `main.py` to launch the user interface.
4. Select an audio file to classify using the provided interface.
5. Click the "Run" button to initiate the classification process.
6. The predicted class label (AI-generated or human-generated) will be displayed.

## File Structure
- `main.py`: Main script to run the user interface and initiate classification.
- `feature_extraction.py`: Functions for extracting MFCC features from audio files.
- `model_training.py`: Script for training the SVM classifier.
- `utils.py`: Utility functions for data preparation and model evaluation.
- `svm_model.joblib`: Trained SVM model saved to a file.
- `dataset/`: Directory containing the dataset of audio files (separate folders for AI-generated and human-generated).

## Acknowledgements
- This project utilizes the librosa library for audio feature extraction.
- The scikit-learn library is used for SVM model training and evaluation.
- The tkinter library is used for building the user interface.

## Credits
This project was developed by Bhuvaneshwaran.

## License
This project is licensed under the MIT License.

