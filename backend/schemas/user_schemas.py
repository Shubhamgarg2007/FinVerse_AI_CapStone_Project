# schemas/user_schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional

# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    fullName: str
    income: float
    riskAppetite: str
    primaryGoal: str
    retirementAge: int
    monthlySavings: float
    investmentExperience: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    class Config:
        orm_mode = True # Allows Pydantic to read data from ORM models

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None