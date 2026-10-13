# Height & Weight Prediction using Machine Learning

A Machine Learning project that predicts **weight from height** and **height from weight** using regression models. The project compares multiple algorithms, evaluates their performance, and deploys the selected models through a Streamlit web application.

## About the Project

This project uses the Height-Weight dataset to study the relationship between human height and weight. Multiple regression algorithms are trained and evaluated using R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Cross-Validation RMSE.

Based on the evaluation results:
- **Linear Regression** is selected for Weight Prediction.
- **Polynomial Regression** is selected for Height Prediction.

The trained models are saved in a single `.pkl` file for use in the Streamlit application.

## Features

- Predict weight using height in centimeters.
- Predict height using weight in kilograms.
- Compare nine regression algorithms.
- Evaluate models using multiple performance metrics.
- Visualize dataset distributions and the relationship between height and weight.
- Display predictions through an interactive Streamlit interface.
- Load pre-trained models from a single PKL file.

## Technologies Used

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
- **Web Framework:** Streamlit
- **Dataset:** Height-Weight Dataset from Kaggle

## Dataset

Dataset source: [Kaggle – Weight-Height Dataset](https://www.kaggle.com/datasets/mustafaali96/weight-height)

The dataset contains the following attributes:

| Column | Description |
|---|---|
| Gender | Gender category |
| Height | Height in inches |
| Weight | Weight in pounds |
| Height_cm | Converted height in centimeters |
| Weight_kg | Converted weight in kilograms |

### Data Preprocessing

- Removed duplicate and missing records.
- Standardized column names.
- Converted height from inches to centimeters.
- Converted weight from pounds to kilograms.
- Removed records containing non-positive height or weight values.
- Applied numerical and categorical preprocessing through Scikit-learn pipelines.

## Machine Learning Models

Nine regression algorithms were compared for both prediction tasks.

1. Linear Regression
2. Polynomial Regression
3. Ridge Regression
4. Lasso Regression
5. Decision Tree Regressor
6. Random Forest Regressor
7. Support Vector Regression (SVR)
8. K-Nearest Neighbors (KNN) Regressor
9. Gradient Boosting Regressor

## Model Evaluation

The following metrics were used to compare model performance:

- **R² Score:** Measures how well the model explains variation in the target variable. Higher is better.
- **MAE:** Measures the average absolute difference between actual and predicted values. Lower is better.
- **RMSE:** Measures prediction error while penalizing larger errors. Lower is better.
- **CV RMSE:** Measures RMSE across cross-validation folds to assess model generalization. Lower is better.

## Results

### Selected Model: Weight Prediction

**Algorithm:** Linear Regression

| Metric | Result |
|---|---:|
| R² Score | 0.9033 |
| MAE | 3.6116 kg |
| RMSE | 4.5641 kg |
| CV RMSE | 4.5376 kg |

The model explains approximately **90.33% of the variation** in weight in the evaluation dataset.

### Selected Model: Height Prediction

**Algorithm:** Polynomial Regression (Degree 2)

| Metric | Result |
|---|---:|
| R² Score | 0.8623 |
| MAE | 2.8889 cm |
| RMSE | 3.6331 cm |
| CV RMSE | 3.6486 cm |

The model explains approximately **86.23% of the variation** in height in the evaluation dataset.

### Model Comparison Summary

| Prediction Task | Selected Model | R² Score |
|---|---|---:|
| Weight Prediction | Linear Regression | 0.9033 |
| Height Prediction | Polynomial Regression | 0.8623 |

The selected models achieved strong results on the evaluation data. For weight prediction, Linear Regression and Ridge Regression produced nearly identical results. For height prediction, Polynomial Regression achieved the highest R² score among the tested models, with a slightly lower RMSE than Linear Regression.

## Project Structure

```text
Height-Weight-Prediction/
│
├── app.py
├── height_weight_models.pkl
├── weight-height.csv
├── requirements.txt
└── README.md
```

- `app.py` – Streamlit application for predictions and visualizations.
- `height_weight_models.pkl` – Contains the trained models and their associated metadata.
- `weight-height.csv` – Dataset used for analysis and visualizations.
- `requirements.txt` – Required Python libraries.
- `README.md` – Project documentation.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SharGit02/Height-Weight_Prediction_SI-LAB-B3_B2_LAB7B.git
cd Height-Weight_Prediction_SI-LAB-B3_B2_LAB7B
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## How to Use

1. Open the Streamlit application.
2. Select **Predict Weight** or **Predict Height**.
3. Enter the required height in centimeters or weight in kilograms.
4. Submit the input to generate a prediction.
5. View the predicted result and available visualizations.

## Visualizations

The application can include the following visualizations:

- Height versus weight scatter plot, grouped by gender.
- Height and weight distribution plots.
- Regression curve generated using the selected model.
- Model comparison charts using the actual evaluation results.

## Saving and Loading Models

The selected models are stored together in one PKL file using Joblib.

```python
import joblib

joblib.dump(best_models, "height_weight_models.pkl")

models = joblib.load("height_weight_models.pkl")

weight_model = models["Weight Prediction"]["model"]
height_model = models["Height Prediction"]["model"]
```

## Limitations

- Predictions are based on patterns learned from the available dataset.
- The dataset contains only height, weight, and gender; other factors affecting body measurements are not considered.
- Predictions may be less reliable for inputs outside the dataset's observed range.
- This project is intended for educational and demonstration purposes, not medical assessment.

## Future Enhancements

- Add additional relevant features to improve prediction quality.
- Explore hyperparameter tuning and additional regression algorithms.
- Deploy the application publicly using Streamlit Community Cloud.
- Improve the user interface with interactive charts and downloadable prediction results.

## Conclusion

This project demonstrates how regression algorithms can be applied to predict human height and weight. Nine models were compared using multiple evaluation metrics, and Linear Regression and Polynomial Regression were selected for the respective prediction tasks. The trained models can be integrated into a Streamlit application to provide interactive predictions and visualizations.

---

**Author: Sharvari Hiwarkhedkar**

Roll No: 23; Class and Batch: B3_B2
