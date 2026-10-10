"""Request / response models for the prediction API."""
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Patient health data (UCI Cleveland feature definitions)."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=260, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=7, description="3 = normal, 6 = fixed defect, 7 = reversible defect")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="P(heart disease)")
    model_name: str
    model_version: str

    model_config = ConfigDict(protected_namespaces=())


class BatchRequest(BaseModel):
    patients: List[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
