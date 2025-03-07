# Air Quality Prediction in Urban Areas

## Overview
This project is a Streamlit-based web application that predicts PM2.5 levels based on air quality and meteorological factors. The model is trained using a Random Forest Regressor and evaluates the impact of different features on air quality.

## Features
- Upload a CSV file containing air quality data.
- Display an overview of the dataset.
- Train a Random Forest model to predict PM2.5 levels.
- Evaluate model performance using Mean Absolute Error (MAE) and R-squared Score.
- Display feature importance using a bar chart.
- Accept user input to predict PM2.5 levels using a trained model.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries.

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the Streamlit app using the following command:
```bash
streamlit run app.py
```

## File Structure
- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of dependencies required for the project.
- `air_quality_model.pkl`: Trained Random Forest model (generated after training).

## Data Format
The uploaded CSV file should contain the following columns:
- `Traffic_Volume`
- `Temperature`
- `Humidity`
- `Wind_Speed`
- `NO2`
- `CO`
- `PM10`
- `PM2.5` (target variable)

## Model Training & Evaluation
- Standardizes feature values using `StandardScaler`.
- Splits data into training and testing sets (80/20 ratio).
- Trains a `RandomForestRegressor` model with 100 estimators.
- Saves the trained model as `air_quality_model.pkl` using `joblib`.
- Evaluates the model using MAE and R-squared score.

## Prediction
- Accepts user inputs for air quality parameters.
- Uses the trained model to predict PM2.5 levels.
- Displays the predicted PM2.5 concentration.

## Dependencies
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `scikit-learn`

## Contributing
Contributions are welcome! Feel free to submit a pull request with enhancements or bug fixes.

## License
This project is licensed under the MIT License.

