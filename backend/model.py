# backend/model.py

import joblib
import pandas as pd

# Load the entire pipeline object. 
# Make sure the filename matches what your training script saves.
# I'll assume the RandomForest model was chosen as best.
pipeline_path = "model_randomforest.pkl" 
pipeline = joblib.load(pipeline_path)

def predict_allocation(user_data: dict) -> dict:
    """
    Takes a dictionary of user data, uses the trained pipeline to preprocess
    and predict, and returns the formatted allocation.
    
    The pipeline handles all necessary scaling and one-hot encoding internally.
    """
    # 1. Convert the single dictionary of user data into a DataFrame.
    # The pipeline's preprocessor expects a DataFrame as input.
    input_df = pd.DataFrame([user_data])
    
    # 2. Use the pipeline to predict. It will automatically:
    #    a. Select the correct columns.
    #    b. Apply StandardScaler to numerical features.
    #    c. Apply OneHotEncoder to categorical features.
    #    d. Pass the processed data to the RandomForestRegressor.
    prediction = pipeline.predict(input_df)[0]
    
    # 3. Format and return the result.
    return {
        'debt_allocation': round(float(prediction[0]), 2),
        'equity_allocation': round(float(prediction[1]), 2),
        'mutual_fund_allocation': round(float(prediction[2]), 2)
    }