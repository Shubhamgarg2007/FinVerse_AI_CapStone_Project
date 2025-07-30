# api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from schemas import user_schemas
from crud import user_crud
from core import security
from db.database import get_db
from jose import JWTError, jwt 

router = APIRouter(

    tags=["Authentication"]
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")



async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # --- START DEBUGGING PRINTS ---
    print("\n--- GATEKEEPER: get_current_user fired ---")
    print(f"--- Received Token: {token[:30]}... ---") # Print first 30 chars of token
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        
        payload = jwt.decode(token, security.settings.SECRET_KEY, algorithms=[security.settings.ALGORITHM])
        email: str = payload.get("sub")
        
        print(f"--- DEBUG: Token successfully decoded. Payload 'sub' (email): {email} ---")

        if email is None:
            print("--- ERROR: 'sub' (email) is missing from token payload. ---")
            raise credentials_exception
            
    except JWTError as e:
        print(f"--- CRITICAL ERROR: JWT decoding failed! Error: {e} ---")
        raise credentials_exception
    
    user = user_crud.get_user_by_email(db, email=email)
    
    if user is None:
        print(f"--- ERROR: User '{email}' from token not found in database. ---")
        raise credentials_exception
    
    print(f"--- SUCCESS: User '{user.email}' authenticated. ---")
   
    return user


@router.post("/signup", response_model=user_schemas.User)
def signup(user: user_schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = user_crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return user_crud.create_user(db=db, user=user)


@router.post("/token", response_model=user_schemas.Token)
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = user_crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=user_schemas.User)
async def read_users_me(current_user: user_schemas.User = Depends(get_current_user)):
    """
    Fetch the details for the currently logged-in user.
    """
    return current_user
