# backend/schemas/model_schema.py

from pydantic import BaseModel
from typing import Optional

class UserInput(BaseModel):
    # Make all fields optional. This allows the frontend to send incomplete
    # data (e.g., from a user profile or just a message). Our endpoint logic in main.py
    # is already set up to handle these missing values by applying defaults.
    age: Optional[int] = None
    annual_income: Optional[int] = None
    monthly_savings: Optional[int] = None
    emergency_fund: Optional[int] = None
    risk_appetite: Optional[str] = None
    investment_goal: Optional[str] = None
    existing_investment_pct: Optional[float] = None
    
    # The message is also optional
    message: Optional[str] = None