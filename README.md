# Continuous Factory Process Data Prediction
This project focuses on developing a predictive machine learning model to analyse and forecast continuous factory process data. The model aims to predict a process parameters based on historical sensor readings, ensuring optimized manufacturing performance and reducing operational inefficiencies.

Key Goals:
* Preprocess raw industrial data from a continuous factory process.
* Engineer relevant features to enhance model performance.
* Train and optimize machine learning models, primarily using Random-Forest Regressors.
* Evaluate model performance using Mean Absolute Error method.
* Deploy the trained model for real-time inference.

## Objective
Predicted Parameter:
* Stage1.Output.Measurement14.U.Actual

Predictor Parameters:
* AmbientConditions.AmbientHumidity.U.Actual
* AmbientConditions.AmbientTemperature.U.Actual
* Machine1.RawMaterial.Property1', 'Machine1.RawMaterial.Property2
* Machine1.RawMaterial.Property3', 'Machine1.RawMaterial.Property4
* Machine1.RawMaterialFeederParameter.U.Actual
* Machine1.Zone1Temperature.C.Actual
* Machine1.Zone2Temperature.C.Actual', 'Machine1.MotorAmperage.U.Actual
* Machine1.MotorRPM.C.Actual', 'Machine1.MaterialPressure.U.Actual
* Machine1.MaterialTemperature.U.Actual
* Machine1.ExitZoneTemperature.C.Actual
* Machine2.RawMaterial.Property1', 'Machine2.RawMaterial.Property2
* Machine2.RawMaterial.Property3', 'Machine2.RawMaterial.Property4
* Machine2.RawMaterialFeederParameter.U.Actual
* Machine2.Zone1Temperature.C.Actual
* Machine2.Zone2Temperature.C.Actual', 'Machine2.MotorAmperage.U.Actual
* Machine2.MotorRPM.C.Actual', 'Machine2.MaterialPressure.U.Actual
* Machine2.MaterialTemperature.U.Actual
* Machine2.ExitZoneTemperature.C.Actual
* Machine3.RawMaterial.Property1', 'Machine3.RawMaterial.Property2
* Machine3.RawMaterial.Property3', 'Machine3.RawMaterial.Property4
* Machine3.RawMaterialFeederParameter.U.Actual
* Machine3.Zone1Temperature.C.Actual
* Machine3.Zone2Temperature.C.Actual', 'Machine3.MotorAmperage.U.Actual
* Machine3.MotorRPM.C.Actual', 'Machine3.MaterialPressure.U.Actual
* Machine3.MaterialTemperature.U.Actual
* Machine3.ExitZoneTemperature.C.Actual
* FirstStage.CombinerOperation.Temperature1.U.Actual
* FirstStage.CombinerOperation.Temperature2.U.Actual
* FirstStage.CombinerOperation.Temperature3.C.Actual']

## Model Performance & Evaluation
The model was evaluated using Mean Absolute Error (MAE) and achieved an average MAE of approximately 12.4%. Further evaluations can be conducted using ```model_evaluation.ipynb``` where different model performances are compared.

## Project Structure
The project is structured into the following folders:

```
Predictive Factory Process Modelling/
│
├── Data/
│   ├── continuous_factory_process.csv               # Raw dataset
│   ├── hyperparameter_results.csv                   # Tuning results
│   ├── notes_on_raw_dataset.txt                     # Dataset further info
│   ├── Processed_Data.csv                           # Cleaned dataset
│   ├── X_train_dataset.csv / y_train_dataset.csv    # Training datasets
│   └── X_test_dataset.csv  / y_test_dataset.csv     # Testing datasets
│
├── Models/
│   ├── Final_Model.pkl                              # Final trained model
│   ├── RandomForest_best.pkl                        # Best RF model
│   ├── X_minmax_scaler.pkl / y_minmax_scaler.pkl    # MinMaxScaler object
│
├── Source Code/
│   ├── load_data.ipynb                              # Load raw dataset
│   ├── pre_data_processing.ipynb                    # Processing of raw dataset
│   ├── feature_engineering.ipynb                    # Feature engineering
│   ├── model_training.ipynb                         # Compares various ML models
│   ├── model_tuning.ipynb                           # Fine tunes parameters (using RF model)
│   ├── model_evaluation.ipynb                       # Evaluates model using MAE
│   └── model_inference.ipynb                        # Deployment of final model
```

## Setup Instructions
1) Ensure that you have Python installed, then install dependencies using the provided ```requirements.txt``` file:
   ```
   pip install -r requirements.txt
   ```
   
2) Activate virtual environment, the project uses a virtual environment called ```myenv```. To activate it run the following:
   ```
   python3 -m venv myenv
   source myenv/bin/activate      # for macOS/Linux
   myenv/Scripts/activate         # for Windows
   ```

3) Run jupyter notebook:
   ```
   jupyter notebook
   ```

## How to use the Model
1) Load the model, scalers, and selected features:
   ```
   import joblib
   best_model = joblib.load('Models/Final_Model.pkl')
   scaler_y = joblib.load('Models/y_minmax_scaler.pkl')
   scaler_X = joblib.load('Models/X_minmax_scaler.pkl')
   with open('../Data/selected_features.txt', 'r') as f:
     selected_features = [line.strip() for line in f]
   ```

2) Load your new data and perform feature engineering:
   ```
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import MinMaxScaler

   # Load a new data sample
   X_new = pd.read_csv('YOUR NEW DATA.csv').sample(n=1)

   # Modify features as per the following
   X_new['Ambient_Temp_Humidity_Interaction'] = X_new['AmbientConditions.AmbientHumidity.U.Actual'] * X_new['AmbientConditions.AmbientTemperature.U.Actual']
   X_new['Machine1_RPM_Amperage_Interaction'] = X_new['Machine1.MotorAmperage.U.Actual'] * X_new['Machine1.MotorRPM.C.Actual']
   X_new['Machine2_RPM_Amperage_Interaction'] = X_new['Machine2.MotorAmperage.U.Actual'] * X_new['Machine2.MotorRPM.C.Actual']
   X_new['Machine3_RPM_Amperage_Interaction'] = X_new['Machine3.MotorAmperage.U.Actual'] * X_new['Machine3.MotorRPM.C.Actual']
   X_new['Machine1_Temp_Press_Interaction'] = X_new['Machine1.MaterialPressure.U.Actual'] * X_new['Machine1.MaterialTemperature.U.Actual']
   X_new['Machine2_Temp_Press_Interaction'] = X_new['Machine2.MaterialPressure.U.Actual'] * X_new['Machine2.MaterialTemperature.U.Actual']
   X_new['Machine3_Temp_Press_Interaction'] = X_new['Machine3.MaterialPressure.U.Actual'] * X_new['Machine3.MaterialTemperature.U.Actual']

   # Drop columns not needed to run the model
   X_new = X_new[selected_features]

   # Apply MinMaxScaler to X data
   X_new_scaled = scaler_X.transform(X_new)
   ```

3) Run the model and predict ```Stage1.Output.Measurement14.U.Actual``` values:
   ```
   y_pred_scaled = best_model.predict(X_new_scaled)

   # Convert prediction back to original scale
   y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
   
   print("Predicted Output:", y_pred[0][0])
   ```

## Contributors
Project ownder: Mo Somji
Developers: Mo Somji
Date: 02/03/2025

If you have suggestions or improvements, feel free to contribute


   











   


















