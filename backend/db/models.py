# db/models.py
from sqlalchemy import Column, Integer, String, Numeric, DateTime
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    fullName = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    income = Column(Numeric(10, 2), nullable=False)
    riskAppetite = Column(String(50), nullable=False)
    primaryGoal = Column(String(100), nullable=False)
    retirementAge = Column(Integer, nullable=False)
    monthlySavings = Column(Numeric(10, 2), nullable=False)
    investmentExperience = Column(String(50), nullable=False)
    createdAt = Column(DateTime(timezone=True), server_default=func.now())