from pydantic import BaseModel
from typing import Optional

class UserInput(BaseModel):

    age: Optional[int] = None
    annual_income: Optional[int] = None
    monthly_savings: Optional[int] = None
    emergency_fund: Optional[int] = None
    risk_appetite: Optional[str] = None
    investment_goal: Optional[str] = None
    existing_investment_pct: Optional[float] = None
    
    # The message is optional
    message: Optional[str] = None