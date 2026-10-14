from datetime import datetime, timezone
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CivicTriage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load("civicfix_model.joblib")
db_complaints = []


class ComplaintInput(BaseModel):
    department: str
    description: str
    suburb: str
    ward: str
    latitude: float
    longitude: float


class ComplaintLogInput(BaseModel):
    department: str
    description: str
    suburb: str
    ward: str
    latitude: float
    longitude: float
    priority: str
    status: str = "New"
    location: str | None = None


class ComplaintStatusUpdate(BaseModel):
    status: str


@app.get("/api/complaints")
def get_complaints():
    return db_complaints


@app.put("/api/complaints/{complaint_id}/status")
def update_complaint_status(complaint_id: str, payload: ComplaintStatusUpdate):
    allowed_statuses = {"New", "Assigned", "Resolved"}
    if payload.status not in allowed_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status '{payload.status}'. Allowed values: {sorted(allowed_statuses)}")

    for complaint in db_complaints:
        if complaint.get("id") == complaint_id:
            complaint["status"] = payload.status
            return complaint

    raise HTTPException(status_code=404, detail=f"Complaint {complaint_id} not found")


@app.post("/api/complaints")
def save_complaint(data: ComplaintLogInput):
    complaint_id = f"#C3-{random.randint(1000, 9999)}"
    complaint = {
        "id": complaint_id,
        "department": data.department,
        "description": data.description,
        "suburb": data.suburb,
        "ward": data.ward,
        "location": data.location or f"{data.suburb}, {data.ward}",
        "latitude": data.latitude,
        "longitude": data.longitude,
        "category": data.department,
        "priority": data.priority,
        "status": data.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db_complaints.append(complaint)
    return {"message": f"Complaint {complaint_id} successfully logged to the queue.", "id": complaint_id}


@app.post("/api/triage")
def predict_urgency(data: ComplaintInput):
    try:
        fault_desc = f"{data.department} - {data.description}"
        input_df = pd.DataFrame([{
            "fault_description": fault_desc,
            "Suburb_Clean": data.suburb,
            "Ward": data.ward,
            "Latitude": data.latitude,
            "Longitude": data.longitude,
        }])

        probabilities = pipeline.predict_proba(input_df)[0]
        prob_critical = probabilities[1]

        if prob_critical >= 0.25:
            prediction = 1
            confidence = float(prob_critical * 100)
        else:
            prediction = 0
            confidence = float(probabilities[0] * 100)

        result = {
            "priority": "Critical" if prediction == 1 else "Routine",
            "confidence": round(confidence, 1),
            "explanation": "Immediate safety risk detected based on infrastructure failure keywords." if prediction == 1 else "Matches standard maintenance patterns.",
            "recommended_action": "Review immediately and follow emergency escalation." if prediction == 1 else "Assign to standard departmental queue.",
        }

        complaint = {
            "id": f"#C3-{len(db_complaints) + 1:04d}",
            "description": data.description,
            "category": data.department,
            "location": f"{data.suburb}, {data.ward}",
            "ward": data.ward,
            "priority": result["priority"],
            "status": "New",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence": result["confidence"],
            "recommended_action": result["recommended_action"],
            "explanation": result["explanation"],
        }
        db_complaints.append(complaint)

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc