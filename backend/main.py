from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db import models
from db.database import engine
from api.routers import auth
from model.model import predict_allocation 
from nlp import parse_natural_language
from schemas.model_schema import UserInput

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinVerse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FinVerse AI API"}

@app.post("/predict")
async def predict(input: UserInput):
    try:

        user_data = input.dict(exclude_unset=True)
        print(f"--- [DEBUG] 1. Initial data from frontend: {user_data}")

        parsed_data = {}
        if input.message:
            parsed_data = parse_natural_language(input.message)
            print(f"--- [DEBUG] 2. Data Parsed from NLP: {parsed_data}")

        final_data = user_data.copy()
        final_data.update(parsed_data)
        print(f"--- [DEBUG] 3. Data after merging NLP: {final_data}")

        default_values = {
            "age": 30, "annual_income": 800000, "monthly_savings": 20000,
            "emergency_fund": 100000, "risk_appetite": "Medium",
            "investment_goal": "Wealth Creation", "existing_investment_pct": 0.1
        }
        for key, value in default_values.items():
            if final_data.get(key) is None:
                final_data[key] = value


        final_data.pop('message', None)

        print(f"--- [DEBUG] 4. FINAL DATA SENT TO MODEL: {final_data} ---")

   
        result = predict_allocation(final_data) 
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")