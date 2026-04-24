"""
Face Attendance System - FastAPI (Render Free Compatible)
No face_recognition (handled in Flutter)
"""

import os, uuid, logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    Boolean, ForeignKey, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ---------------- CONFIG ----------------
SECRET_KEY   = os.getenv("SECRET_KEY", "change-this")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60 * 24

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")
PHOTO_DIR    = "student_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ---------------- DATABASE ----------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True)
    name = Column(String)
    hashed_pw = Column(String)
    role = Column(String, default="student")
    photo_path = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DBAttendance(Base):
    __tablename__ = "attendance"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    marked_at = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ---------------- AUTH ----------------
pwd_ctx = CryptContext(schemes=["bcrypt"])
oauth2  = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(p): return pwd_ctx.hash(p)
def verify_password(p, h): return pwd_ctx.verify(p, h)

def create_token(data):
    data["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(DBUser).filter(DBUser.id == payload["sub"]).first()
        if not user: raise
        return user
    except:
        raise HTTPException(401, "Invalid token")

def require_admin(user: DBUser = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    return user

# ---------------- SCHEMAS ----------------
class TokenOut(BaseModel):
    access_token: str
    role: str
    user_id: str
    name: str

class UserCreate(BaseModel):
    email: str
    name: str
    password: str

class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    has_photo: bool

class VerifyResult(BaseModel):
    success: bool
    message: str
    attendance_id: Optional[str] = None

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT ADMIN ----------------
@app.on_event("startup")
def seed():
    db = SessionLocal()
    if not db.query(DBUser).filter(DBUser.role == "admin").first():
        db.add(DBUser(
            email="admin@admin.com",
            name="Admin",
            hashed_pw=hash_password("admin123"),
            role="admin"
        ))
        db.commit()
    db.close()

# ---------------- AUTH ----------------
@app.post("/auth/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"sub": user.id})
    return TokenOut(access_token=token, role=user.role, user_id=user.id, name=user.name)

# ---------------- STUDENTS ----------------
@app.post("/students", response_model=UserOut)
def create_student(data: UserCreate, db: Session = Depends(get_db), _=Depends(require_admin)):
    u = DBUser(
        email=data.email,
        name=data.name,
        hashed_pw=hash_password(data.password)
    )
    db.add(u); db.commit(); db.refresh(u)
    return UserOut(id=u.id, email=u.email, name=u.name, role=u.role, has_photo=False)

@app.get("/students", response_model=List[UserOut])
def list_students(db: Session = Depends(get_db), _=Depends(require_admin)):
    users = db.query(DBUser).filter(DBUser.role=="student").all()
    return [
        UserOut(id=u.id, email=u.email, name=u.name, role=u.role, has_photo=bool(u.photo_path))
        for u in users
    ]

@app.post("/students/{id}/photo")
async def upload_photo(id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    u = db.query(DBUser).filter(DBUser.id == id).first()
    if not u: raise HTTPException(404, "User not found")

    path = f"{PHOTO_DIR}/{id}.jpg"
    with open(path, "wb") as f:
        f.write(await file.read())

    u.photo_path = path
    db.commit()
    return {"msg": "uploaded"}

# ---------------- ATTENDANCE ----------------
@app.post("/attendance/verify", response_model=VerifyResult)
def mark_attendance(
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db),
    user: DBUser = Depends(get_current_user),
):
    today = datetime.utcnow().replace(hour=0, minute=0, second=0)

    existing = db.query(DBAttendance).filter(
        DBAttendance.user_id==user.id,
        DBAttendance.marked_at>=today
    ).first()

    if existing:
        return VerifyResult(success=True, message="Already marked", attendance_id=existing.id)

    rec = DBAttendance(user_id=user.id, latitude=latitude, longitude=longitude)
    db.add(rec); db.commit(); db.refresh(rec)

    return VerifyResult(success=True, message="Attendance marked", attendance_id=rec.id)

# ---------------- HEALTH ----------------
@app.get("/")
def root():
    return {"status": "running"}
