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


# re-engineer bmi
df1["bmi_cat"] = np.where(df1['bmi']<18.5, 'Underweight',
                   np.where(df1['bmi']<25, 'Normal',
                   np.where(df1['bmi']<30, 'Overweight', 'Obese')))

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
df1.loc[:,'bmi_cat'] = label_encoder.fit_transform(df1['bmi_cat'])
print(df1.head())


# make prediction
df2 = df1[['systolic', 'diastolic', 'protein_in_urine', 'temperature', 'bmi_cat', 'blood_for_glucose', 'syphilis']].copy()
model = Predictor()
predictions = model.predict(df2)
print(Counter(predictions))


# applying clasical definition
hdp = (df2["systolic"] >=140) | (df2["diastolic"] >= 90)
df2.loc[:,'label'] = 0
df2.loc[hdp,"label"] = 1
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

# unsupervised classification (also called clustering)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


# get data and process
df3 = df2.drop(['label'], axis='columns')

# Impute missing values for KMeans
imputer = SimpleImputer(strategy='mean')
df3_imputed = imputer.fit_transform(df3)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df3_imputed)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df3_imputed)

print("Cluster Labels:", kmeans.labels_)

# Analyze results
labels = kmeans.labels_

# adding clusters to dataframe
df3["cluster"] = kmeans.labels_
df3["prediction"] = df2["label"]
df3["detection"] = np.where((df3['systolic'] >= 140) | (df3['diastolic'] >= 90), 1, 0)


# check correlation
df3_out = df3[['cluster','prediction','detection']].copy()
df3_corr = df3_out.corr()
print(df3_corr)

# plot correlation
import seaborn as sns
sns.heatmap(df3_corr, annot=True, cmap='coolwarm')
plt.title("Output Correlation Matrix")
plt.savefig('figures/output_correlation.png')
#plt.show()

# save to csv
df3.to_csv(Path(BASE_DIR).joinpath("data/further_analysis.csv"), index=False) 



print(df3.head())
# print("dataframe : ", df3_imputed.shape[0])  # Number of rows in DataFrame
# print("clusters :", len(kmeans.labels_))  # Number of cluster labels
# print(df3_imputed)

plt.scatter(df3_imputed[:, 0], df3_imputed[:, 1], c=labels)
plt.xlabel('Systolic Values')
plt.ylabel('Diastolic Values')
plt.title('Cluster Visualization')
plt.savefig('figures/clustering.png')
#plt.show()


## AUC
## New content added Wed Mar 26th, 2025
from sklearn.metrics import roc_auc_score, roc_curve
X_test = df1[['systolic', 'diastolic', 'protein_in_urine', 'temperature', 'bmi_cat', 'blood_for_glucose', 'syphilis']].copy()
y_test = df2["label"]
y_probs = []
for i in predictions:
    if i=="No Risk":
        y_probs.append(0)
    else:
        y_probs.append(1)

# Compute AUC score
auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.4f}")

## Confidence interval
from sklearn.utils import resample
n_bootstraps = 1000  # Number of bootstrap samples
auc_scores = []
for i in range(n_bootstraps):
    # Resample with replacement
    X_resampled, y_resampled = resample(X_test, y_test, random_state=i)

    # Get predictions for the resampled dataset
    predictions2 = model.predict(X_resampled)
    y_resampled_probs = []
    for i in predictions2:
        if i=="No Risk":
            y_resampled_probs.append(0)
        else:
            y_resampled_probs.append(1)

    # Compute AUC
    auc_bootstrap = roc_auc_score(y_resampled, y_resampled_probs)
    auc_scores.append(auc_bootstrap)

# Compute 95% Confidence Interval
lower = np.percentile(auc_scores, 2.5)
upper = np.percentile(auc_scores, 97.5)

print(f"95% Confidence Interval for AUC: ({lower:.4f}, {upper:.4f})")

# plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", color='blue')

# Plot confidence bands
plt.fill_between(fpr, np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5), 
                 color='blue', alpha=0.2, label="95% CI")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random Guess Line
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve with 95% Confidence Interval")
plt.legend()
#plt.show()
plt.savefig('figures/roc_curve2.png')


# another rock curve plot
# added Nov 15, 2024
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ground truth (numeric 0/1)
y_true = df2['label'].astype(int).values

# obtain continuous scores (preferred)
y_score = None
try:
    # try Predictor.predict_proba(X) -> (n,2) array
    y_score = model.predict_proba(df2)[:, 1]
    print("Using model.predict_proba(...) for scores.")
except Exception:
    try:
        # maybe Predictor exposes internal trained estimator as .clf or .estimator
        y_score = model.clf.predict_proba(df2)[:, 1]
        print("Using model.clf.predict_proba(...) for scores.")
    except Exception:
        # fallback: convert predicted class labels to binary (NOT recommended)
        print("Warning: model does not provide predict_proba. Using discrete predictions as scores.")
        y_score = np.array([1 if p == "Risk exist" or p == 1 else 0 for p in predictions])

# compute AUC and ROC
roc_auc = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
print(f"AUC Score: {roc_auc:.4f}")

# plot ROC
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--', alpha=0.6)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(alpha=0.2)
plt.savefig('figures/roc_curve3.png')
plt.clf()

# confusion matrix at threshold (choose threshold as needed)
th = 0.5
y_pred_th = (y_score >= th).astype(int)

cm = confusion_matrix(y_true, y_pred_th)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix at threshold {th}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    cm_display = np.array([[tn, fp],
                           [fn, tp]])
    annot = np.array([["TN\n{:d}".format(tn), "FP\n{:d}".format(fp)],
                      ["FN\n{:d}".format(fn), "TP\n{:d}".format(tp)]])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_display, annot=annot, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix (threshold={th})')
    plt.savefig('figures/confusion_matrix_model_test.png')
    plt.clf()
else:
    print("Multiclass confusion matrix:")
    print(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig('figures/confusion_matrix_multiclass_model_test.png')
    plt.clf()

# optional: show classification report using thresholded predictions
print("Classification report at threshold", th)
print(classification_report(y_true, y_pred_th, target_names=["No Risk","Risk exist"]))

# analysis of the discordant pairs
# November 6th, 2025
from scipy.stats import ttest_ind, chi2_contingency
# predictions is the array/list returned by Predictor().predict(df2)
pred_labels = np.array([1 if str(p).lower().startswith('risk') else 0 for p in predictions])
y_true = df2['label'].astype(int).values

# try to get continuous scores if Predictor supports predict_proba
y_score = None
try:
    y_score = model.predict_proba(df2)[:, 1]
    print("Using continuous scores from predict_proba.")
except Exception:
    print("predict_proba not available; proceeding with discrete predictions only.")

# --- confusion summary ---
cm = confusion_matrix(y_true, pred_labels)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None,)*4
print("Confusion matrix (tn, fp, fn, tp):", (tn, fp, fn, tp))
print(classification_report(y_true, pred_labels, target_names=['NoRisk', 'Risk']))

# --- masks for concordance/discordance ---
tp_mask = (y_true == 1) & (pred_labels == 1)
tn_mask = (y_true == 0) & (pred_labels == 0)
fp_mask = (y_true == 0) & (pred_labels == 1)   # model predicts risk but conventional says no
fn_mask = (y_true == 1) & (pred_labels == 0)   # model misses conventional risk

# save discordant cases for manual review
discordant_df = df2.loc[fp_mask | fn_mask].copy()
discordant_df['model_pred'] = pred_labels[fp_mask | fn_mask]
discordant_df['true_label'] = y_true[fp_mask | fn_mask]
discordant_df.to_csv(Path(BASE_DIR).joinpath('model/discordant_cases.csv'), index=False)
print(f"Saved {len(discordant_df)} discordant cases to model/discordant_cases.csv")

# --- descriptive comparisons ---
df2_eval = df2.copy()
df2_eval['pred'] = pred_labels
df2_eval['pair_type'] = np.where(tp_mask, 'TP', np.where(tn_mask, 'TN', np.where(fp_mask, 'FP', 'FN')))

# numeric summaries for key features (adjust list as needed)
numeric_feats = ['systolic','diastolic','bmi_cat','blood_for_glucose','temperature','weight','height'] 
numeric_feats = [c for c in numeric_feats if c in df2_eval.columns]

summary_by_pair = df2_eval.groupby('pair_type')[numeric_feats].agg(['count','mean','std']).round(3)
summary_by_pair.to_csv(Path(BASE_DIR).joinpath('model/discordant_summary.csv'), index=True)
print("Numeric feature summary by pair type:")
print(summary_by_pair)

# quick t-tests between concordant (TP+TN) and discordant (FP+FN) for each numeric feature
concordant_mask = tp_mask | tn_mask
discordant_mask = fp_mask | fn_mask
print("\nT-tests (concordant vs discordant) for numeric features:")
for col in numeric_feats:
    a = pd.to_numeric(df2_eval.loc[concordant_mask, col], errors='coerce').dropna()
    b = pd.to_numeric(df2_eval.loc[discordant_mask, col], errors='coerce').dropna()
    if len(a) >= 2 and len(b) >= 2:
        stat, p = ttest_ind(a, b, equal_var=False)
        print(f"{col}: n_concord={len(a)}, n_disc={len(b)}, t={stat:.3f}, p={p:.3e}")
    else:
        print(f"{col}: insufficient data for t-test")

# categorical features: contingency tables + chi-square
cat_feats = ['protein_in_urine','glucose_in_urine','blood_group','syphilis'] 
cat_feats = [c for c in cat_feats if c in df2_eval.columns]
print("\nCategorical feature chi-square (concordant vs discordant):")
pair_group = df2_eval['pair_type'].apply(lambda x: 'concordant' if x in ('TP','TN') else 'discordant')
pair_group = pd.Categorical(pair_group, categories=['concordant','discordant'])

for col in cat_feats:
    # ensure feature is string / categorical before fillna to avoid downcast warning
    feat = df2_eval[col].astype(str).fillna('NA')
    tbl = pd.crosstab(feat, pair_group)
    if tbl.size == 0:
        continue
    try:
        chi2, p, dof, exp = chi2_contingency(tbl)
        print(f"{col}: chi2={chi2:.3f}, p={p:.3e}, dof={dof}")
    except Exception as e:
        print(f"{col}: chi2 error {e}")

# --- score distributions for discordant groups (if y_score available) ---
if y_score is not None:
    df2_eval['score'] = y_score
    plt.figure(figsize=(8,4))
    sns.kdeplot(df2_eval.loc[tp_mask, 'score'], label='TP', bw_adjust=1)
    sns.kdeplot(df2_eval.loc[tn_mask, 'score'], label='TN', bw_adjust=1)
    sns.kdeplot(df2_eval.loc[fp_mask, 'score'], label='FP', bw_adjust=1)
    sns.kdeplot(df2_eval.loc[fn_mask, 'score'], label='FN', bw_adjust=1)
    plt.legend()
    plt.title('Predicted score distribution by pair type')
    plt.xlabel('Predicted score (probability)')
    plt.savefig(Path(BASE_DIR).joinpath('figures/score_dist_by_pair.png'))
    plt.clf()

# --- SHAP inspection for discordant cases (optional) ---
try:
    import shap
    # get a small sample of discordant indices for inspection
    sample_idx = discordant_df.index[:6].tolist()
    X_sample = df2.loc[sample_idx, numeric_feats]  # adjust features to the training feature set used for SHAP
    explainer = shap.TreeExplainer(model.clf if hasattr(model,'clf') else model)  # adjust to underlying estimator if available
    shap_vals = explainer.shap_values(X_sample)
    # plot waterfall for first discordant example (pick correct indexing for multi-class)
    try:
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.savefig(Path(BASE_DIR).joinpath('figures/shap_discordant_example.png'))
        plt.clf()
    except Exception:
        pass
except Exception as e:
    print("SHAP inspection skipped:", e)