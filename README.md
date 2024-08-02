# Flower Classification CNN Model

This project implements a Convolutional Neural Network (CNN) model for classifying images of flowers into five categories: daisy, dandelion, rose, sunflower, and tulip.

## Features

- Train a CNN model on a dataset of flower images
- Classify flower images using the trained model
- Real-time classification using webcam input
- Streamlit web application for easy interaction

## Installation

1. Clone this repository:
git clone (https://github.com/ennajari/Floral_Recognition)
cd <repository-directory>

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

## Usage
---> . To run the Streamlit web application:
streamlit run app.py

## Project Structure

- `app.py`: Main Streamlit application for flower classification
- `Flower_Recog_Model.keras`: Saved trained model
- `Images/`: Directory containing the dataset of flower images
- `Sample/`: Directory containing sample images for testing

## Model Performance

The model achieves high accuracy on both training and validation sets. Refer to the generated plots for detailed performance metrics.

## Contributors

Ennajari abdellah
