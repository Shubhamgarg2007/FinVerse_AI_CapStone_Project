# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db import models
from db.database import engine
from api.routers import auth

# This line creates the 'users' table in your MySQL database if it doesn't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinVerse AI API")

# Configure CORS
origins = [
    "http://localhost:5173", # Your React frontend
    "http://localhost:3000", # Common alternative for React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the authentication router
app.include_router(auth.router, prefix="/api")
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the FinVerse AI API"}