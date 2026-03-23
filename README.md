# Phone Price Prediction

This project predicts mobile phone prices using a Linear Regression model.

## Project Description

The project includes training a Linear Regression model, saving it with pickle, making predictions from terminal input, and running a simple Flask web interface that displays a graph, metrics, and dataset samples.

## Libraries

- pandas
- scikit-learn
- matplotlib
- flask

## How to Run

Install libraries:
pip install -r requirements.txt

Run the project:
python train.py
python predict.py
python app.py

Then open in browser:
http://127.0.0.1:5000

## Preprocessing

- removed Unnamed: 0 column because it is only an index
- converted Brand me into numeric dummy columns
- converted numeric columns using pd.to_numeric(errors='coerce')
- removed missing or invalid values using dropna()

## Metrics

The model is evaluated using:
- R2 Score
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

## Files

- train.py — model training
- predict.py — terminal prediction
- app.py — web application
- templates/index.html — web page
- static/model_graph.png — graph
- requirements.txt — dependencies

## Notes

The model shows moderate performance because Linear Regression does not fully capture complex relationships in the dataset.