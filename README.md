# MMLINDE Mama (MMama) Prediction Model
MLINDE MAMA (MMama) Prediction Model is a machine learning model trained on routine antenatal visit data predict negative maternal outcomes. The model input seven parameters which are 1. systolic, 2.diastolic, 3.gestational age, 4.protein in urine, 5. temperature, 6.bmi and 7.blood for glucose and output a binary indicator for risk or no-risk of certain conditions. 

Click [here](https://ai.phit.or.tz) to view the online demo

## Prerequisites
> - Python 3.8+

## Installation instructions
1. clone the repository

```
git clone https://github.com/ilyatuu/mmama
```

2. change into the directory
```
cd mmama
```
3. install virtual environment
```
python3 -m venv venv
```
4. activate virtual environment
```
source venv/bin/activate
```
5. install requirements
```
pip install -r requirements.txt
```
6. launch the server
```
uvicorn main:app --reload
```

## Usage
Open browser and connect to localhost using ``port 8000``.
```
127.0.0.1:8000
```

Send a json payload using the following ``curl`` command. Data is submitted in a json array ``[{...},{...},...]``
```
curl -X POST "127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d '[{"systolic":120,"diastolic":78,"gestationage":29,"protein_in_urine":0,"temperature":37,"bmi":24.44,"blood_for_glucose":5.7},{"systolic":110,"diastolic":80,"gestationage":20,"protein_in_urine":1,"temperature":37,"bmi":22.44,"blood_for_glucose":5.6}]'
```