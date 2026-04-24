"""
Face Attendance System - Single File FastAPI Backend
Install: pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] face_recognition pillow python-multipart sqlalchemy aiofiles
Run: uvicorn main:app --reload
"""

import os, io, base64, json, uuid, logging
from datetime import datetime, timedelta
from typing import Optional, List

import face_recognition
import numpy as np
from PIL import Image

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    Boolean, Text, ForeignKey, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production-use-long-random-string")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60 * 24  # minutes (1 day)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")
PHOTO_DIR    = "student_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


class DBUser(Base):
    __tablename__ = "users"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email        = Column(String, unique=True, index=True, nullable=False)
    name         = Column(String, nullable=False)
    hashed_pw    = Column(String, nullable=False)
    role         = Column(String, default="student")   # "student" | "admin"
    photo_path   = Column(String, nullable=True)        # path to reference face image
    face_encoding= Column(Text, nullable=True)          # JSON array of 128 floats
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    attendances  = relationship("DBAttendance", back_populates="user")


class DBAttendance(Base):
    __tablename__ = "attendance"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    marked_at  = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, nullable=True)  # face match distance (lower = better)
    latitude   = Column(Float, nullable=True)
    longitude  = Column(Float, nullable=True)
    user       = relationship("DBUser", back_populates="attendances")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2  = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_token(data: dict, expires_minutes: int = TOKEN_EXPIRE) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> DBUser:
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Bad token payload")
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_admin(user: DBUser = Depends(get_current_user)) -> DBUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# ─────────────────────────────────────────────
# FACE HELPERS
# ─────────────────────────────────────────────
def image_from_upload(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def get_encoding(img_array: np.ndarray) -> Optional[List[float]]:
    locs     = face_recognition.face_locations(img_array)
    encodings = face_recognition.face_encodings(img_array, locs)
    if not encodings:
        return None
    return encodings[0].tolist()


def compare_faces(stored_enc: List[float], live_enc: List[float], threshold: float = 0.50) -> tuple[bool, float]:
    distance = face_recognition.face_distance(
        [np.array(stored_enc)], np.array(live_enc)
    )[0]
    return bool(distance < threshold), float(distance)


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str
    user_id: str


class UserCreate(BaseModel):
    email: str
    name: str
    password: str
    role: str = "student"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    has_photo: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AttendanceOut(BaseModel):
    id: str
    user_id: str
    student_name: str
    marked_at: datetime
    confidence: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]


class AttendanceStats(BaseModel):
    total_students: int
    present_today: int
    absent_today: int
    total_records: int


class VerifyResult(BaseModel):
    success: bool
    message: str
    confidence: Optional[float] = None
    attendance_id: Optional[str] = None


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="Face Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Seed default admin on first run ──────────
@app.on_event("startup")
def seed_admin():
    db = SessionLocal()
    try:
        if not db.query(DBUser).filter(DBUser.role == "admin").first():
            admin = DBUser(
                email="admin@school.com",
                name="Admin",
                hashed_pw=hash_password("admin123"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            log.info("Default admin created: admin@school.com / admin123")
    finally:
        db.close()


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/auth/login", response_model=TokenOut, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    token = create_token({"sub": user.id, "role": user.role})
    return TokenOut(access_token=token, role=user.role, name=user.name, user_id=user.id)


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
def me(user: DBUser = Depends(get_current_user)):
    return UserOut(
        id=user.id, email=user.email, name=user.name,
        role=user.role, has_photo=bool(user.photo_path),
        is_active=user.is_active, created_at=user.created_at
    )


# ─────────────────────────────────────────────
# STUDENT ROUTES (admin manages students)
# ─────────────────────────────────────────────
@app.get("/students", response_model=List[UserOut], tags=["Students"])
def list_students(db: Session = Depends(get_db), _=Depends(require_admin)):
    users = db.query(DBUser).filter(DBUser.role == "student").all()
    return [
        UserOut(
            id=u.id, email=u.email, name=u.name,
            role=u.role, has_photo=bool(u.photo_path),
            is_active=u.is_active, created_at=u.created_at
        ) for u in users
    ]


@app.post("/students", response_model=UserOut, tags=["Students"])
def create_student(body: UserCreate, db: Session = Depends(get_db), _=Depends(require_admin)):
    if db.query(DBUser).filter(DBUser.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    u = DBUser(
        email=body.email, name=body.name,
        hashed_pw=hash_password(body.password),
        role="student",
    )
    db.add(u); db.commit(); db.refresh(u)
    return UserOut(
        id=u.id, email=u.email, name=u.name,
        role=u.role, has_photo=False,
        is_active=u.is_active, created_at=u.created_at
    )


@app.delete("/students/{student_id}", tags=["Students"])
def delete_student(student_id: str, db: Session = Depends(get_db), _=Depends(require_admin)):
    u = db.query(DBUser).filter(DBUser.id == student_id, DBUser.role == "student").first()
    if not u:
        raise HTTPException(status_code=404, detail="Student not found")
    db.delete(u); db.commit()
    return {"message": "Deleted"}


@app.post("/students/{student_id}/photo", tags=["Students"])
async def upload_reference_photo(
    student_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _=Depends(require_admin),
):
    u = db.query(DBUser).filter(DBUser.id == student_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Student not found")

    raw = await file.read()
    img = image_from_upload(raw)
    enc = get_encoding(img)
    if enc is None:
        raise HTTPException(status_code=422, detail="No face detected in uploaded photo")

    # Save image file
    ext  = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    path = os.path.join(PHOTO_DIR, f"{student_id}.{ext}")
    with open(path, "wb") as f:
        f.write(raw)

    u.photo_path    = path
    u.face_encoding = json.dumps(enc)
    db.commit()
    return {"message": "Reference photo saved and face encoded"}


@app.get("/students/{student_id}/photo", tags=["Students"])
def get_student_photo(student_id: str, db: Session = Depends(get_db), _=Depends(get_current_user)):
    u = db.query(DBUser).filter(DBUser.id == student_id).first()
    if not u or not u.photo_path or not os.path.exists(u.photo_path):
        raise HTTPException(status_code=404, detail="Photo not found")
    ext = u.photo_path.split(".")[-1].lower()
    mt  = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return StreamingResponse(open(u.photo_path, "rb"), media_type=mt)


# ─────────────────────────────────────────────
# ATTENDANCE — VERIFY & MARK (Student)
# ─────────────────────────────────────────────
@app.post("/attendance/verify", response_model=VerifyResult, tags=["Attendance"])
async def verify_and_mark(
    file: UploadFile = File(...),
    latitude:  Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Students only")

    if not current_user.face_encoding:
        raise HTTPException(status_code=422, detail="No reference face registered. Ask admin to upload your photo.")

    raw = await file.read()
    img = image_from_upload(raw)
    live_enc = get_encoding(img)
    if live_enc is None:
        return VerifyResult(success=False, message="No face detected in selfie. Try better lighting.")

    stored_enc = json.loads(current_user.face_encoding)
    matched, distance = compare_faces(stored_enc, live_enc)

    if not matched:
        return VerifyResult(
            success=False,
            message=f"Face not recognised (distance={distance:.2f}). Try again.",
            confidence=distance,
        )

    # Prevent duplicate within same day
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing = (
        db.query(DBAttendance)
        .filter(DBAttendance.user_id == current_user.id, DBAttendance.marked_at >= today_start)
        .first()
    )
    if existing:
        return VerifyResult(
            success=True,
            message="Already marked present today.",
            confidence=distance,
            attendance_id=existing.id,
        )

    record = DBAttendance(
        user_id=current_user.id,
        confidence=distance,
        latitude=latitude,
        longitude=longitude,
    )
    db.add(record); db.commit(); db.refresh(record)

    return VerifyResult(
        success=True,
        message="Attendance marked successfully!",
        confidence=distance,
        attendance_id=record.id,
    )


# ─────────────────────────────────────────────
# ATTENDANCE — READ (Admin)
# ─────────────────────────────────────────────
@app.get("/attendance", response_model=List[AttendanceOut], tags=["Attendance"])
def get_attendance(
    date: Optional[str] = None,   # "YYYY-MM-DD"
    student_id: Optional[str] = None,
    db: Session = Depends(get_db),
    _=Depends(require_admin),
):
    q = db.query(DBAttendance).join(DBUser)
    if date:
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            q = q.filter(
                DBAttendance.marked_at >= d,
                DBAttendance.marked_at < d + timedelta(days=1),
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    if student_id:
        q = q.filter(DBAttendance.user_id == student_id)

    rows = q.order_by(DBAttendance.marked_at.desc()).all()
    return [
        AttendanceOut(
            id=r.id,
            user_id=r.user_id,
            student_name=r.user.name,
            marked_at=r.marked_at,
            confidence=r.confidence,
            latitude=r.latitude,
            longitude=r.longitude,
        )
        for r in rows
    ]


@app.get("/attendance/my", response_model=List[AttendanceOut], tags=["Attendance"])
def my_attendance(
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    rows = (
        db.query(DBAttendance)
        .filter(DBAttendance.user_id == current_user.id)
        .order_by(DBAttendance.marked_at.desc())
        .all()
    )
    return [
        AttendanceOut(
            id=r.id,
            user_id=r.user_id,
            student_name=current_user.name,
            marked_at=r.marked_at,
            confidence=r.confidence,
            latitude=r.latitude,
            longitude=r.longitude,
        )
        for r in rows
    ]


@app.get("/attendance/stats", response_model=AttendanceStats, tags=["Attendance"])
def attendance_stats(db: Session = Depends(get_db), _=Depends(require_admin)):
    total_students = db.query(DBUser).filter(DBUser.role == "student").count()
    today_start    = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    present_today  = (
        db.query(func.count(func.distinct(DBAttendance.user_id)))
        .filter(DBAttendance.marked_at >= today_start)
        .scalar()
    )
    total_records  = db.query(DBAttendance).count()
    return AttendanceStats(
        total_students=total_students,
        present_today=present_today,
        absent_today=total_students - present_today,
        total_records=total_records,
    )


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Face Attendance API running"}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
