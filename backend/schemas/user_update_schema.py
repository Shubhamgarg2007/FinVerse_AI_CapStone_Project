from pydantic import BaseModel, EmailStr
from typing import Optional

class UserUpdate(BaseModel):
    fullName: Optional[str] = None
    email: Optional[EmailStr] = None
    income: Optional[float] = None
    riskAppetite: Optional[str] = None
    primaryGoal: Optional[str] = None
    retirementAge: Optional[int] = None
    monthlySavings: Optional[float] = None
    investmentExperience: Optional[str] = None