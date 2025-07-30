import joblib
import pandas as pd


pipeline_path = "model_randomforest.pkl" 
pipeline = joblib.load(pipeline_path)

def predict_allocation(user_data: dict) -> dict:
    """
    Takes a dictionary of user data, uses the trained pipeline to preprocess
    and predict, and returns the formatted allocation.
    
    The pipeline handles all necessary scaling and one-hot encoding internally.
    """
    
    input_df = pd.DataFrame([user_data])
    
  
    prediction = pipeline.predict(input_df)[0]
    
    
    return {
        'debt_allocation': round(float(prediction[0]), 2),
        'equity_allocation': round(float(prediction[1]), 2),
        'mutual_fund_allocation': round(float(prediction[2]), 2)
    }