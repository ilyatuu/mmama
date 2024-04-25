# MMLINDE Mama (MMama) Prediction Model
MLINDE MAMA (MMama) Prediction Model is a machine learning model trained on routine antenatal visit data predict negative maternal outcomes. The model input seven parameters which are 1. systolic, 2.diastolic, 3.gestational age, 4.protein in urine, 5. temperature, 6.bmi and 7.blood for glucose and output a binary indicator for risk or no-risk of certain conditions. 

##### Prerequisites
> - Python 3.8+

##### Installation instructions
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