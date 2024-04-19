from pydantic import BaseModel

class ItemIn(BaseModel):
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