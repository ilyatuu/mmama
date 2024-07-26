import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.metrics import accuracy_score, classification_report
from model_create import Predictor

BASE_DIR = Path(__file__).resolve(strict=True).parent
df0 = pd.read_csv( Path(BASE_DIR).joinpath("data/anc_test_data2.csv") , low_memory = False)


# data cleaning
df0.loc[df0['gest_age']=='020', 'gest_age'] = 20
df0.loc[df0['gest_age']=='334', 'gest_age'] = 33
df0.loc[df0['gest_age']=='30w 5days', 'gest_age'] = 30
df0.loc[df0['gest_age']=='17 weeks and 7 days', 'gest_age'] = 17
df0.loc[df0["protein_in_urine"]=="negative, [negative]", "protein_in_urine"] = "negative"
df0.loc[df0["glucose_in_urine"]=="negative, [negative]", "glucose_in_urine"] = "negative"
df0.loc[df0["blood_for_glucose"]=="5.7, [5.7]", "blood_for_glucose"] = "5.7"
df0.loc[df0["blood_group"]=="O, [O]", "blood_group"] = "O"

# cleaning visit number
df0.loc[df0["visit_number"]=="2, [2]", "visit_number"] = "2"
df0.loc[df0["visit_number"]=="3, [3]", "visit_number"] = "3"
df0.loc[df0["visit_number"]=="4, [4]", "visit_number"] = "4"
df0.loc[df0["visit_number"]=="5, [5]", "visit_number"] = "5"
df0.loc[df0["visit_number"]=="6, [6]", "visit_number"] = "6"
df0.loc[df0["visit_number"]=="7, [7]", "visit_number"] = "7"
df0.loc[df0["visit_number"]=="9, [9]", "visit_number"] = "7"

# if different, take the smallest number
df0.loc[df0["visit_number"]=="2, [3]", "visit_number"] = "2"
df0.loc[df0["visit_number"]=="3, [4]", "visit_number"] = "3"
df0.loc[df0["visit_number"]=="2, [3]", "visit_number"] = "2"
df0.loc[df0["visit_number"]=="3, [2]", "visit_number"] = "2"
df0.loc[df0["visit_number"]=="4, [5]", "visit_number"] = "4"
df0.loc[df0["visit_number"]=="6, [7]", "visit_number"] = "6"
df0.loc[df0["visit_number"]=="4, [2]", "visit_number"] = "2"
df0.loc[df0["visit_number"]=="5, [6]", "visit_number"] = "5"

df0.loc[df0["height"]=="66, [66]", "height"] = "66"
df0.loc[df0["height"]=="77, [77]", "height"] = "77"
df0.loc[df0["height"]=="125, [125]", "height"] = "125"
df0.loc[df0["height"]=="196, [196]", "height"] = "196"
df0.loc[df0["height"]=="152.8, [152.8]", "height"] = "152.8"
df0.loc[df0["height"]=="156.9, [156.9]", "height"] = "156.9"
df0.loc[df0["height"]=="162.3, [162.3]", "height"] = "162.3"
df0.loc[df0["height"]=="163.4, [163.4]", "height"] = "163.4"
df0.loc[df0["height"]=="151.1, [151.1]", "height"] = "151.1"
df0.loc[df0["height"]=="168.3, [168.3]", "height"] = "151.1"
df0.loc[df0["height"]=="163.7, [163.7]", "height"] = "163.7"
df0.loc[df0["height"]=="150.8, [150.8]", "height"] = "150.8"
df0.loc[df0["height"]=="143.5, [143.5]", "height"] = "143.5"
df0.loc[df0["height"]=="165.4, [165.4]", "height"] = "165.4"
df0.loc[df0["height"]=="158.8, [158.8]", "height"] = "158.8"


# global changes
df0 = df0.replace('[null]', np.nan)
df0 = df0.sort_values(['client_id','visit_number'],ascending = [False, True])

# change column data types
df0['bmi'] = pd.to_numeric(df0['bmi'], errors='coerce')
df0['weight'] = pd.to_numeric(df0['weight'], errors='coerce')
df0['height'] = pd.to_numeric(df0['height'], errors='coerce')
df0['gest_age'] = pd.to_numeric(df0['gest_age'], errors='coerce')
df0['systolic'] = pd.to_numeric(df0['systolic'], errors='coerce')
df0['diastolic'] = pd.to_numeric(df0['diastolic'], errors='coerce')
df0['temperature'] = pd.to_numeric(df0['temperature'], errors='coerce')
df0['visit_number'] = pd.to_numeric(df0['visit_number'], errors='coerce')
df0['blood_for_glucose'] = pd.to_numeric(df0['blood_for_glucose'], errors='coerce')


#print(df0['gest_age'].dtype)
#print(df0['weight'].value_counts())

df1 = df0.groupby('client_id',as_index = False
                      ).agg(
                              {
                                  'gest_age':'max',
                                  'glucose_in_urine':'last',
                                  'protein_in_urine':'last',
                                  'blood_group':'last','syphilis':'last',
                                  'visit_number':'last',
                                  'blood_for_glucose':'mean',
                                  'height':'mean',
                                  'weight':'mean',
                                  'bmi':'mean',
                                  'systolic':'mean',
                                  'diastolic':'mean',
                                  'temperature':'mean'
                                }
                              )


print('The shape of data before collapsing:', df0.shape)
print('The shape of data after collapsing:', df1.shape)
print('Maximum number of visit per client:')
print(df1['visit_number'].value_counts().sort_index(ascending=True))

print(df1.isna().sum()/len(df1)*100)
print(df1.head())


# protein_in_urine_cat = []
# for row in df1['protein_in_urine']:
#     if row == 'test_not_conducted':
#         protein_in_urine_cat.append(2)
#     elif row == 'negative':
#         protein_in_urine_cat.append(0)
#     else:
#         protein_in_urine_cat.append(1)

# df1.loc[:,'protein_in_urine'] = protein_in_urine_cat


# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the column to label encoded values
df1.loc[:,'syphilis'] = label_encoder.fit_transform(df1['syphilis'])
df1.loc[:,'blood_group'] = label_encoder.fit_transform(df1['blood_group'])
df1.loc[:,'glucose_in_urine'] = label_encoder.fit_transform(df1['glucose_in_urine'])
df1.loc[:,'protein_in_urine'] = label_encoder.fit_transform(df1['protein_in_urine'])
print(df1.head())


# make prediction
df2 = df1[['systolic', 'diastolic', 'protein_in_urine', 'temperature', 'bmi', 'blood_for_glucose', 'syphilis']].copy()
model = Predictor()
predictions = model.predict(df2)
print(Counter(predictions))


# applying clasical definition
hddp = (df2["systolic"] >=140) | (df2["diastolic"] >= 90)
df2.loc[:,'label'] = 0
df2.loc[hddp,"label"] = 1
print(df2["label"].value_counts())

labels = []
# apply labels
for i in df2['label']:
    if i==0:
        labels.append("No Risk")
    else:
        labels.append("Risk exist")
labels = np.asarray(labels)
print(labels)

# check accuracy
accuracy = accuracy_score(labels, predictions)
print("Prediction accuracy: %.2f%%" % (accuracy * 100.0))

# classification report
cr = classification_report(labels, predictions)
print('Classification report :')
print(cr)