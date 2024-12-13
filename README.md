# Emotion detector
The Emotion Detector project allows you to detect emotions from images you upload. It utilizes a trained model to classify emotions with high accuracy.

## Features
- **Emotion Classification**: Upload an image, and the app identifies the emotion displayed.
- **Customizable Training**: Adjust hyperparameters to fine-tune the model for better performance.

![photo](https://user-images.githubusercontent.com/64987384/110686353-644f8800-81f0-11eb-9d14-b5da09a5296d.jpg)

## Installation
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. Download the training dataset and required files:
   - [Training Dataset](https://drive.google.com/file/d/1TG9P5B2k3eTbC4XDxDmEc07dyAORPC16/view)
   - [Dataframe File with Targets](https://www.kaggle.com/c/skillbox-computer-vision-project/data?select=train.csv)

## Usage

1) **Train the Model**: Run the following command to train the model:
   ```bash
   python train.py
   ```
2) **Run the Application**: After training the model, start the application with:
    ```bash
    python run.py
3) Upload an image through the app interface and see the detected emotion.

## Customization
  - Edit `params.py` to adjust settings such as image size, batch size, model name, path to an image, etc.
