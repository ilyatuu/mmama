from pydantic import BaseModel
from typing import Optional

class ItemIn(BaseModel):
    id:Optional[int]=999
    systolic: float
    diastolic: float
    gestationage: float
    protein_in_urine:int
    temperature:float
    bmi:float
    blood_for_glucose:float

class ItemOut(BaseModel):
    systolic: float
    diastolic: float
    gestationage: float
    protein_in_urine:int
    temperature:float
    bmi:float
    blood_for_glucose:float
    prediction:str
    id:int