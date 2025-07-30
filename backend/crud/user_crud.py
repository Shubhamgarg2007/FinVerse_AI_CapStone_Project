# crud/user_crud.py
from sqlalchemy.orm import Session
from db import models
from schemas import user_schemas
from core.security import get_password_hash
from schemas import user_schemas, user_update_schema

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: user_schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        fullName=user.fullName,
        income=user.income,
        riskAppetite=user.riskAppetite,
        primaryGoal=user.primaryGoal,
        retirementAge=user.retirementAge,
        monthlySavings=user.monthlySavings,
        investmentExperience=user.investmentExperience,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

