# Bug Prediction App

## Overview
The **Bug Prediction App** is a machine learning-based web application built using *Streamlit*. It predicts potential bugs in a system based on user feedback and the company name. This project leverages a **RandomForestClassifier** to classify the type of bug that might be associated with specific user feedback, enabling developers to quickly identify and address potential issues in their systems.

## Features
- **User-Friendly Interface**: The app provides an easy-to-use interface where users can input the company name and user feedback to predict the associated bug.
- **Bug Prediction**: The app predicts the likely bug based on the input data using a trained Random Forest classifier.
- **Visualization**: The app provides a confusion matrix and a bug frequency bar chart to help users understand the model's performance and the distribution of different bugs.

## Dataset
The dataset used in this project is generated randomly and consists of the following features:

- **Company Name**: The name of the company where the bug was reported.
- **Bug Name**: The type of bug that was identified.
- **User Feedback**: The feedback provided by the user that is associated with the bug.

## Dependencies
To run this project, you need to install the following dependencies:

- Python 3.x
- pandas
- scikit-learn
- streamlit
- matplotlib
- seaborn

You can install the required packages using pip:

pip install pandas scikit-learn streamlit matplotlib seaborn
## How to Run the App

Clone the repository to your local machine:

git clone https://github.com/yourusername/bug-prediction-app.git
cd bug-prediction-app
Install the required dependencies:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run bug_prediction_app.py
The app will open in your web browser. You can input the company name and user feedback to get the predicted bug.

## Project Structure

- **bug_prediction_app.py**: The main Python script that contains the code for the Streamlit app and the machine learning model.
- **README.md**: The file you are currently reading.
- **requirements.txt**: The file that contains the list of dependencies needed to run the app.

## Model Details

- **RandomForestClassifier**: The model used for predicting the bug types. It is trained on a dataset of simulated data with company names, user feedback, and associated bugs.

## Visualizations

The app provides the following visualizations:

- **Confusion Matrix**: Displays the performance of the model in predicting the correct bugs.
- **Bug Frequency Bar Chart**: Shows the frequency of each bug in the dataset.

## Future Enhancements

- Integrate a real-world dataset for more accurate predictions.
- Implement additional machine learning models for better performance.
- Enhance the user interface for a better user experience.
- Add more features such as bug severity prediction.
