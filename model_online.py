import pandas as pd
import numpy as np
import itertools
import ast
import joblib
import shap
from pathlib import Path

from time import time
from random import randint
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost.sklearn import  XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion,Pipeline


BASE_DIR = Path(__file__).resolve(strict=True).parent

anc_visits = pd.read_csv( Path(BASE_DIR).joinpath("data/anc_visits.csv") , low_memory = False)

##reduce dataset to key variables
data1 = anc_visits[['client_id','ganc','gest_age','blood_for_glucose','glucose_in_urine',\
                    'protein_in_urine','blood_group','syphilis','rh_factor','visit_number',\
                        'height','weight','bmi','systolic','diastolic','temperature','fundal_height','fetal_heart_rate']]

#check for null
data1.isna().sum()/len(data1)*100
#or
total = data1.isnull().sum().sort_values(ascending=False)
percent = round(data1.isnull().sum()/data1.isnull().count()*100, 2).sort_values(ascending = False)
summary = pd.concat([total,percent],axis = 1, keys = ['Total', 'Percent']).transpose()

# visit frequency table
vs = data1['visit_number'].value_counts()


# collapse to individual observation using client id and visit number
data2 = data1.sort_values(['client_id','visit_number'],ascending = [False, True])
data3 = data2.replace('[null]', np.nan)
data4 = data3.groupby('client_id',as_index = False).agg({
                                    'ganc':'last',
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
                                    'temperature':'mean'})

print('Data shape before collapsing:', data2.shape)
print('Data shape after collapsing:', data4.shape)
print('Number of clients per visits  after collapsing:', data4['visit_number'].value_counts())
print('Number of GANC clients per visits:', data4['ganc'].value_counts())


# re-engineer data
# create categories for bmi
data4["bmi_cat"] = np.where(data4['bmi']<18.5, 'Underweight',
                   np.where(data4['bmi']<25, 'Normal',
                   np.where(data4['bmi']<30, 'Overweight', 'Obese')))


#separating categorical and numerical variables
categorical_attributes = ['ganc','protein_in_urine','glucose_in_urine','blood_group','syphilis', 'bmi_cat']
numerical_attributes = [i for i in data4.columns if i not in categorical_attributes]



#for individual cross-section analysis : individual observation
data4_cat = data4[categorical_attributes]
data4_num  = data4[numerical_attributes]
#drop ganc and client_id, we don't need them pass this point
#data4_cat = data4_cat.drop(['ganc'], axis = 1)
data4_num = data4_num.drop(['client_id', 'bmi'], axis = 1)

#create a label encoder function
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler
def labelencoder(X, y = 0):
    encoded_df = pd.DataFrame()
    encoder = LabelEncoder()
    for i in X.columns:
        encoded =  encoder.fit_transform(X[i])
        encoded = pd.DataFrame(
                                data = encoded,
                                columns = [i]
                              )

        encoded_df = pd.concat(
                                [
                                    encoded_df,
                                    encoded
                                ],
                                axis = 1
                              )

    return encoded_df

# use the label encoder
data4_cat_enco = labelencoder(data4_cat)

# combine dataframe
data5 = pd.concat([data4_cat_enco,data4_num],axis = 1)

# deriving the outcome variable
# derive hypertensive dissorders during pregnancy hdp using 
# classical definitions
hdp = (data5['systolic'] >= 140) | (data5['diastolic'] >= 90)

data5['hdp'] = 0              # No Risk of hdp
data5.loc[hdp, 'hdp'] = 1    # Risk of hdp

# print data labeling
print('Print the outcome condition (hypertensive disorder during pregnancy - hdp) using cutppoints')
print('0:No-Risk, 1: Risk of hdp \n', data5['hdp'].value_counts())

# reduce the dataset then drop NA
data6 = data5[['systolic', 'diastolic', 'protein_in_urine', 'temperature', 'bmi_cat', 'blood_for_glucose', \
               'syphilis', 'hdp']]
print('label count before dropping NA:', data6['hdp'].value_counts())

# index data
label_2_rows = data6[data6['hdp'] == 1]

# fill NA with median value
median_values = label_2_rows.drop('hdp', axis = 1).median()
for index, row in label_2_rows.iterrows():
    data6.loc[index] = data6.loc[index].fillna(median_values)

data6_no_NAs = data6.dropna() # this to be discussed, and presented for discussion
print('label count after dropping NA:', data6_no_NAs['hdp'].value_counts())

training_data = data6_no_NAs.loc[(data6_no_NAs['hdp'] == 0) | (data6_no_NAs['hdp'] == 1)]
print('training data', training_data['hdp'].value_counts())


## chi2 test for categorical variables
## Added november 15th, 2025
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2 as sk_chi2
from sklearn.preprocessing import MinMaxScaler

# Use training_data (created earlier) which contains encoded categorical columns and numeric features
X_fs = training_data.drop(columns=['hdp']).copy()
y_fs = training_data['hdp'].astype(int).copy()

# Identify categorical vs numeric columns (adjust if your list differs)
categorical_cols_fs = [c for c in categorical_attributes if c in X_fs.columns]
numeric_cols_fs = [c for c in X_fs.columns if c not in categorical_cols_fs]

# 1) Chi-square on categorical features (contingency table + chi2 test + Cramer's V effect size)
chi2_results = []
n = len(X_fs)
for col in categorical_cols_fs:
    ct = pd.crosstab(X_fs[col], y_fs)
    # If a category has only one column in the crosstab (rare), skip
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        chi2_results.append({'feature': col, 'chi2': np.nan, 'p_value': np.nan, 'cramers_v': np.nan})
        continue
    chi2_stat, p, dof, expected = chi2_contingency(ct)
    r, k = ct.shape
    denom = n * (min(r - 1, k - 1))
    cramers_v = np.sqrt(chi2_stat / denom) if denom > 0 else np.nan
    chi2_results.append({'feature': col, 'chi2': chi2_stat, 'p_value': p, 'cramers_v': cramers_v})

chi2_df = pd.DataFrame(chi2_results).set_index('feature').sort_values('p_value')

# 2) Mutual information for all features (works for mixed types once encoded as integers)
# mutual_info_classif expects numeric array; X_fs is already numeric (label-encoded cats)
X_mi = X_fs.copy()
mi_scores = mutual_info_classif(X_mi, y_fs, discrete_features=[c in categorical_cols_fs for c in X_mi.columns], random_state=42)
mi_df = pd.Series(mi_scores, index=X_mi.columns).sort_values(ascending=False).to_frame(name='mutual_info')

# 3) Combine results into a single table for review
feature_scores = pd.concat([chi2_df, mi_df], axis=1).fillna(np.nan)
feature_scores = feature_scores.sort_values(by=['mutual_info', 'cramers_v'], ascending=False)
feature_scores.to_csv('model/feature_selection_scores.csv')
print("Feature selection scores (top 10):")
print(feature_scores.head(10).to_string())

# 4) Example: SelectKBest usage
# chi2 requires non-negative features -> scale numeric features to [0,1] for chi2
scaler_for_chi2 = MinMaxScaler()
X_for_chi2 = pd.DataFrame(scaler_for_chi2.fit_transform(X_fs), columns=X_fs.columns)

k = min(5, X_for_chi2.shape[1])
skb_chi2 = SelectKBest(sk_chi2, k=k).fit(X_for_chi2, y_fs)
selected_chi2 = X_for_chi2.columns[skb_chi2.get_support()].tolist()

skb_mi = SelectKBest(mutual_info_classif, k=k).fit(X_fs, y_fs)
selected_mi = X_fs.columns[skb_mi.get_support()].tolist()

print(f"Top {k} by chi2 (SelectKBest): {selected_chi2}")
print(f"Top {k} by mutual information (SelectKBest): {selected_mi}")

# 5) Interpretation hints (print summary)
print("\nInterpretation:")
print("- chi2 p_value small => association between categorical feature and target (but effect size also matters)")
print("- Cramer's V gives effect size (0..1). Higher = stronger association.")
print("- mutual_info measures non-linear / general dependence; higher is better.")
print("- Use these scores together (and model-based importance) to choose robust features.")
## end of chi2 test


# Rule of thumb balance the instances, when one class is overepresented
# Here we will undersample the majority class to match the minority class
# The aim is to avoid bias

# define X (matrix of features) and y (list of labels)

X = training_data.iloc[:, :-1] # select all columns except the last one
y = training_data["hdp"]

# Balance the training data
# return x-resampled and y-resampled
rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)
print('The number of samples after resampling:', Counter(y_res))

# Concatinate resampled data for explorations by plots

temp_df = pd.concat([X_res,y_res],axis = 1).reset_index(drop = True)

# Drawing figures
# 1. Normal distribution curve
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest, probplot
mean = data5['gest_age'].mean()
std_dev = data5['gest_age'].std()
bin_size = 30
max_gest_age = 60

# new variables, new test
gest = data5['gest_age'].dropna()
gest_clean = gest[gest.between(0, max_gest_age)]  # remove outliers beyond 0-80 weeks
N = len(gest)
median = gest.median()
mode = gest.mode().iloc[0] if not gest.mode().empty else np.nan
skewness = gest.skew()
kurtosis = gest.kurtosis()


print(f"n={N}, mean={mean:.3f}, median={median:.3f}, mode={mode}, std={std_dev:.3f}, skew={skewness:.3f}, kurtosis={kurtosis:.3f}")

# --- Normality test ---
# Shapiro for small samples, D'Agostino (normaltest) for larger samples
if N <= 5000:
    stat, p = shapiro(gest)
    test_name = "Shapiro-Wilk"
else:
    stat, p = normaltest(gest)
    test_name = "D'Agostino K^2"

print(f"{test_name}: stat={stat:.4f}, p={p:.4f}  (p>0.05 suggests no evidence against normality)")

# QQ-plot
plt.figure(figsize=(6,6))
probplot(gest, dist="norm", plot=plt)
plt.title("QQ-plot of gestational age")
plt.savefig('figures/qqplot_gest_age.png')
plt.clf()


## Counting values within a target gestational age
target_age = 25 # the mean gestational age is 25.168
# 1) exact count
exact_count = data5['gest_age'].eq(target_age).sum()
print(f"Exact gest_age == {target_age}: {exact_count}")

# 2) rounded to the nearest week
rounded_count = data5['gest_age'].round().eq(target_age).sum()
print(f"gest_age rounded == {target_age}: {rounded_count}")

# 3) tolerance window (recommended): e.g. ±0.5 weeks
tol = 0.5
mask_close = data5['gest_age'].between(mean - tol, mean + tol)
count_within_tol = mask_close.sum()
print(f"Observations within ±{tol} of mean ({mean:.2f}): {count_within_tol}")

# Frequency by visit_number for those observations (if visit_number exists in data5)
if 'visit_number' in data5.columns:
    df_close = data5.loc[mask_close, ['visit_number', 'gest_age']].copy()
    visits_counts = df_close['visit_number'].value_counts().sort_index()
    print("Counts by visit_number for observations near mean:")
    print(visits_counts.to_string())
else:
    print("visit_number not present in data5; cannot show visit breakdown.")

# count using tolerance window (what you already computed)
tol_count = gest.between(target_age - tol, target_age + tol).sum()
counts, bin_edges = np.histogram(gest, bins=bin_size)
bin_width = bin_edges[1] - bin_edges[0]

# find which bin contains target_age
bin_index = np.searchsorted(bin_edges, target_age, side='right') - 1
if bin_index < 0 or bin_index >= len(counts):
    bin_count = 0
else:
    bin_count = counts[bin_index]

print(f"Total non-NA observations: {N}")
print(f"Count within ±{tol} weeks of {target_age}: {tol_count}")
print(f"Histogram bins={bin_size}, bin_width={bin_width:.3f} weeks")
print(f"Bin containing {target_age}: {bin_edges[bin_index]:.3f} to {bin_edges[bin_index+1]:.3f}")
print(f"Count in that histogram bin: {bin_count}")

# create 1-week bins so each bar is ±0.5 around an integer week
week_bins = np.arange(gest_clean.min() - 0.5, gest_clean.max() + 0.5 + 1e-6, 1.0)
bin_width = week_bins[1] - week_bins[0]
N_clean = len(gest_clean)

x_clean = np.linspace(gest_clean.min(), gest_clean.max(), 1000)
normal_clean = norm.pdf(x_clean, mean, std_dev)

# scale pdf to histogram counts: counts ≈ pdf * N * bin_width
scaled_normal= normal_clean * N_clean * bin_width

plt.hist(gest_clean, bins=week_bins, density=False, facecolor='#666666', edgecolor='black', linewidth=0.8, label='Gestational Age')
plt.plot(x_clean, scaled_normal, color='black', linestyle='--', label='Normal Curve')
plt.xlim(0, max_gest_age)
plt.title('Distribution of Gestational Age (1-week bins)')
plt.xlabel('Gestational Age')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('figures/normal_freq2.png')
plt.clf()   # clear figure
### end

rows_near_target = data5.loc[mask_close]
#print(f"Example rows (up to 5):\n{rows_near_target.head().to_string()}")



# Plot the histogram
plt.hist(gest, bins=bin_size, density=True, alpha=0.6, color='blue', label='Histogram (density)')

# Plot the normal distribution curve
# Generate the normal distribution curve
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
normal_curve = norm.pdf(x, mean, std_dev)
plt.plot(x, normal_curve, color='red', label='Normal Curve')

# Add labels and legend
plt.title('Gestation Age Distribution with Normal Curve')
plt.xlabel('Gestational Age')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('figures/normal_density.png')
#plt.show()
plt.clf()   # clear figure

# repeat the normal curve with frequency values
# to do this, remove density=True. This makes the histogram display absolute frequencies instead of normalized densities
normal_freq = norm.pdf(x, mean, std_dev) * len(data5['gest_age']) * (max(data5['gest_age']) - min(data5['gest_age'])) / bin_size

# Plot the histogram with thinner bars and white borders
plt.hist(data5['gest_age'], bins=bin_size, edgecolor='white', alpha=0.6, color='blue', label='Data Histogram')
#bin_count = 30  # Number of bins
#n, bins, patches = plt.hist(data5['gest_age'], bins=bin_count, alpha=0.6, color='blue', edgecolor='white', linewidth=0.7, label='Data Histogram', rwidth=0.5)
plt.plot(x, normal_freq, color='red', label='Normal Curve')

# Format the y-axis with commas
from matplotlib.ticker import FuncFormatter
def format_with_commas(x, pos):
    return f"{int(x):,}"  # Format the tick as an integer with commas
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_commas))

# Rescale x-axis
plt.xlim(0, 100)

#plt.title('Histogram with Normal Distribution Curve (Frequency)')
plt.xlabel('Gestation Age')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('figures/normal_freq.png')
plt.clf()   # clear figure


# violin plot
import seaborn as sns
sns.set_theme(context = "paper", style = "white", palette = "deep", font_scale = 2.0, color_codes = True, rc = ({'font.family': 'Dejavu Sans'}))
plt.figure(figsize = (15, 15))
fig, axs = plt.subplots(1, 3)

sns.violinplot(x = "hdp",y = "systolic",hue = 'hdp',data = temp_df,palette = 'colorblind',ax = axs[0], legend = False)
sns.violinplot(x = "hdp",y = "diastolic",hue = 'hdp',data = temp_df,palette = 'colorblind',ax = axs[1], legend = False)
sns.violinplot(x = "hdp",y = "temperature",hue = 'hdp',data = temp_df,palette = 'colorblind',ax = axs[2],legend = False)
fig.tight_layout(pad = .5)
fig.savefig('figures/violine.png')

plt.rcParams["figure.figsize"] = [15,15]
sns.pairplot(temp_df, hue = "hdp", height = 3, palette = 'colorblind').savefig('figures/scatterplots.png')


plt.clf()

plt.rcParams["figure.figsize"] = [15,15]
sns.heatmap(temp_df.corr(method = 'pearson'),annot = True,vmin = -1,vmax = 1).get_figure().savefig('figures/heatmap1.png')
# clear figure
plt.clf()

# drop protein in urine as it has no influence in the model
# corr = temp_df.drop(['protein_in_urine'], axis = 1).corr(method = 'pearson')
corr = temp_df.corr(method = 'pearson')

# the above line does not give p-values
# write a custom function to return p-values on correlation

# Function to calculate correlations and p-values
from scipy.stats import pearsonr
def correlation_with_pvalues(df):
    cols = df.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols)
    pval_matrix = pd.DataFrame(index=cols, columns=cols)

    for i in cols:
        for j in cols:
            corr, pval = pearsonr(df[i], df[j])
            corr_matrix.loc[i, j] = corr
            pval_matrix.loc[i, j] = pval

    return corr_matrix, pval_matrix

corr2, p_values = correlation_with_pvalues(temp_df.drop(['protein_in_urine'], axis = 1))
print("Pearson Correlation Matrix:")
print(corr2)

print("\nP-value Matrix:")
print(p_values)

# create the mask to remote the top part of the correlation matrix
#mask = np.zeros_like(corr, dtype=bool)
#mask[np.tril_indices_from(mask)] = True
#np.fill_diagonal(mask,False)
mask = np.triu(np.ones_like(corr,dtype=bool))

# Consider showing p-value in the correlation matrix
# see the link below
# https://tosinharold.medium.com/enhancing-correlation-matrix-heatmap-plots-with-p-values-in-python-41bac6a7fd77

#sns.set_theme(font_scale=1.8)
sns.heatmap(corr,
            annot = True,
            vmin = -1,
            vmax = 1,
            fmt='.2f',
            mask=mask).get_figure().savefig('figures/heatmap2.png')
#sns.heatmap(temp_df.drop(['protein_in_urine'], axis = 1).corr(method = 'pearson'),annot = True,vmin = -1,vmax = 1).get_figure().savefig('figures/heatmap2.png')
# Split training and test data
# We wont drop glucose in urine as for now, until we have discussed and agreed
from sklearn.model_selection import (GridSearchCV,train_test_split,cross_val_score,RandomizedSearchCV,KFold)
X_res_temp = np.asarray(X_res)
y_res_temp = np.asarray(y_res)


# split train sets and test sets
X_train, X_test, y_train, y_test  = train_test_split(X_res_temp,y_res_temp,test_size = 0.1,shuffle = True,random_state = 42)

print('The shape of X train : {}'.format(X_train.shape))
print('The shape of y train : {}'.format(y_train.shape))
print('The shape of X test : {}'.format(X_test.shape))
print('The shape of y test : {}'.format(y_test.shape))

# scale data

# Its fine to apply StandardScaler to these features even if they have different magnitudes.
# The scaler will standardize each feature independently into mean = 0, and SD=1, so it doesn't matter if some features
# have larger values than others (i.e. systolic & diastolic).

scaler = StandardScaler().fit(X = X_train)
scl_features = scaler.transform(X = X_train)

# save scaler to disk
joblib.dump(scaler, 'model/mmama_scaler.joblib')

## Model definition
# define parameters

num_folds = 5 # split data into five folds
seed = np.random.randint(0, 81470) # random seed value
scoring = 'accuracy' # metric for model evaluation

# specify cross-validation strategy
kf = KFold(n_splits = num_folds,shuffle = True,random_state = seed)

# make a list of models to test
models = []
models.append(('KNN', KNeighborsClassifier()))
#models.append(('LR', LogisticRegression(multi_class = 'ovr',max_iter = 2000,random_state = seed)))
models.append(('LR', LogisticRegression(max_iter = 2000,random_state = seed)))
models.append(('SVM', SVC(kernel = 'linear',gamma = 'auto',random_state = seed)))
models.append(('RF', RandomForestClassifier(n_estimators = 500,random_state = seed)))
models.append(('XGB', XGBClassifier(random_state = seed,n_estimators = 500)))

# Feature selection
# Using mlextend algorithm
# define temporary model
temp_model = XGBClassifier(random_state = seed,n_estimators = 500)

# Sequential Forward Floating Selection
sffs = SFS(
              temp_model,
              # k_features = (2, 7),  # Selecting the "best" feature combination in a k-range, f k_features is set to to a tuple (min_k, max_k)
              k_features = 4,forward = True,floating = True,scoring = 'accuracy',cv = kf,n_jobs = -1)

# make a pipeline
pipe = Pipeline([('scaler', StandardScaler()),('sffs', sffs)])

# fit


# fit algorithm
pipesfs = pipe.fit(X_res, y_res)
# retrieve the best combination information from the last step of the pipeline (sffs)
best_combination_acc = pipe.named_steps['sffs'].k_score_
best_combination_idx = pipe.named_steps['sffs'].k_feature_idx_

print('best combination (ACC: %.3f): %s\n' % (best_combination_acc, best_combination_idx))
# print('all subsets:\n', sffs.subsets_)

# Plotting
plt.rcParams["figure.figsize"] = [6, 6]
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(pipe.named_steps['sffs'].get_metric_dict(),kind = 'std_err')
plt.title('Sequential Forward Selection')
plt.savefig('figures/plot_sequential.png')

# Feature selection
# Using recursive future elimination algorithm

# Create an RFE selector with cross-validation
rfecv = RFECV(estimator = temp_model,step = 1,min_features_to_select = 4,cv = kf,scoring = 'accuracy')

# make a pipeline
pipe = Pipeline([('scaler', StandardScaler()),('rfecv', rfecv)])

# Fit the RFE selector on your data
pipe.fit(X_res, y_res)

# Get the selected feature indices
selected_features_indices = pipe.named_steps['rfecv'].support_

# Get the names of the final features
final_feature_names = X_res.columns[selected_features_indices]

# Print the final selected features
print("Selected Features:", final_feature_names)

# Evaluate models to get the best perfoming model
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate
from scipy.stats import t as t_dist
results = []
names = []
summary_stats = []

# scorers to compute for each model
scoring_dict = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

for name, model in models:
    names.append(name)

    # build an imblearn pipeline that applies SMOTE only to the training folds
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=seed)),
        ('clf', model)
    ])

    # run cross_validate to get all metrics per-fold (test scores)
    cv_res = cross_validate(
        estimator=pipeline,
        X=scl_features,
        y=y_train,
        cv=kf,
        scoring=scoring_dict,
        return_train_score=False,
        n_jobs=-1
    )

    # collect accuracy array for plotting compatibility with earlier code
    acc_array = cv_res['test_accuracy']
    results.append(acc_array)

    # compute mean, sd and 95% CI for each metric and print
    model_summary = {'model': name}
    for metric in scoring_dict.keys():
        key = f'test_{metric}'
        vals = cv_res.get(key)
        if vals is None:
            mean_val = np.nan
            std_val = np.nan
            lower = upper = np.nan
        else:
            vals = np.asarray(vals, dtype=float)
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals, ddof=1)
            n_cv = vals.size
            se = std_val / np.sqrt(n_cv) if n_cv > 0 else 0.0
            ci_mult = t_dist.ppf(1 - 0.025, df=n_cv - 1) if n_cv > 1 else 0.0
            ci95 = ci_mult * se
            # clamp for proportion metrics
            lower = max(0.0, mean_val - ci95)
            upper = min(1.0, mean_val + ci95)

        # print one line per metric
        print(f"{name} | {metric}: mean={mean_val:.4%}, std={std_val:.4%}, 95% CI=({lower:.4%}, {upper:.4%})")

        model_summary[f'{metric}_mean'] = mean_val
        model_summary[f'{metric}_std'] = std_val
        model_summary[f'{metric}_ci95_lower'] = lower
        model_summary[f'{metric}_ci95_upper'] = upper

    summary_stats.append(model_summary)


# save summary stats to CSV
summary_df = pd.DataFrame(summary_stats).set_index('model')
summary_df.to_csv('model/cv_summary_with_ci.csv')

## No need, this is printed in the looop
# print("\nCross-validation summary with 95% CI:")
# print(summary_df.to_string())


# prepare results_df for the existing boxplot (accuracy)
results_df = pd.DataFrame(results,columns = (0, 1, 2, 3, 4)).T # columns should correspond to the number of folds, k = 5
results_df.columns = names
results_df = pd.melt(results_df) # melt data frame into a long format.
results_df.rename(columns = {'variable':'Model', 'value':'Accuracy'},inplace = True)
# print("Cross Validaiton Results")
# print(results_df)

# Plotting the algorithm selection
plt.figure(figsize = (6, 4))
plt.clf()
sns.boxplot(x = 'Model',y = 'Accuracy',data = results_df,hue = 'Model',legend = False,palette = 'colorblind')
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .1, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ")
plt.title("Cross Validation Scores")
plt.tight_layout()
plt.savefig('figures/classification_comparison.png')


# big LOOP for ML training
# TUNNING THE SELECTED MODEL

from sklearn.metrics import roc_curve,auc,accuracy_score,roc_auc_score,confusion_matrix,classification_report,precision_recall_fscore_support
# Set validation procedure
num_folds = 5 # split training set into 5 parts for validation
num_rounds = 10 # increase this to 5 or 10 once code is bug-free
seed = 42 # pick any integer. This ensures reproducibility of the tests
random_seed = np.random.randint(0, 81478)

features = training_data.iloc[:, :-1]
scoring = 'accuracy' # score model accuracy

# cross validation strategy
kf = KFold(
            n_splits = num_folds,
            shuffle = True,
            random_state = random_seed
          )

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

save_predicted, save_true = [], [] # save predicted and true values for each loop
start = time()

# Specify model
# Define the XGBoost model for training and prediction
classifier = XGBClassifier()

# set hyparameter

estimators = [500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

random_grid = {'n_estimators': estimators,'learning_rate': rate,'max_depth': depth,'min_child_weight': child_weight,'gamma': gamma,'colsample_bytree': bytree}


for round in range (num_rounds):

  # cross validation and splitting of the validation set

  for train_index, test_index in kf.split(scl_features, y_train):
      X_train_set, X_val_set = scl_features[train_index], scl_features[test_index]
      y_train_set, y_val = y_train[train_index], y_train[test_index]

    #   print('The shape of X train set : {}'.format(X_train_set.shape))
    #   print('The shape of y train set : {}'.format(y_train_set.shape))
    #   print('The shape of X validation : {}'.format(X_val_set.shape))
    #   print('The shape of y validation : {}'.format(y_val.shape))

      # generate models using all combinations of settings

      # RANDOMSED GRID SEARCH
      # Random search of parameters, using 5 fold cross validation,
      # search across 100 different combinations, and use all available cores

      n_iter_search = 10
      rsCV = RandomizedSearchCV(
                                  verbose = 1,
                                  estimator = classifier,
                                  param_distributions = random_grid,
                                  n_iter = n_iter_search,
                                  scoring = scoring,
                                  cv = kf,
                                  refit = True,
                                  n_jobs = -1
                              )

      rsCV_result = rsCV.fit(X_train_set, y_train_set)

      # print out results and give hyperparameter settings for best one
      means = rsCV_result.cv_results_['mean_test_score']
      stds = rsCV_result.cv_results_['std_test_score']
      params = rsCV_result.cv_results_['params']

    #   for mean, stdev, param in zip(means, stds, params):
    #       print("%.2f (%.2f) with: %r" % (mean, stdev, param))

      # print best parameter settings
      print("Best: %.2f using %s" % (rsCV_result.best_score_,rsCV_result.best_params_))

      # Insert the best parameters identified by randomized grid search into the base classifier
      best_classifier = classifier.set_params(**rsCV_result.best_params_)

      # Fit your models
      best_classifier.fit(X_train_set, y_train_set)

      # predict test instances
      predictions = best_classifier.predict(X_val_set)

      # zip all predictions for plotting averaged confusion matrix

      for predicted, true in zip(predictions, y_val):
          save_predicted.append(predicted)
          save_true.append(true)

      # local confusion matrix & classification report
      local_cm = confusion_matrix(y_val, predictions)
      local_report = classification_report(y_val, predictions)

      # append feauture importances
      local_feat_impces = pd.DataFrame(
                                                best_classifier.feature_importances_,
                                                index = features.columns
                                              ).sort_values(
                                                              by = 0,
                                                              ascending = False
                                                            )

      # summarizing results
      local_kf_results = pd.DataFrame(
                                        [
                                            ("Accuracy", accuracy_score(y_val, predictions)),
                                            ("TRAIN",str(train_index)),
                                            ("TEST",str(test_index)),
                                            ("CM", local_cm),
                                            ("Classification report", local_report),
                                            ("y_test", y_val),
                                            ("Feature importances", local_feat_impces.to_dict())
                                        ]
                                    ).T

      local_kf_results.columns = local_kf_results.iloc[0]
      local_kf_results = local_kf_results[1:]
      kf_results = pd.concat(
                              [
                                  kf_results,
                                  local_kf_results
                              ],
                              axis = 0,
                              join = 'outer'
                            ).reset_index(drop = True)

      # per class accuracy
      local_support = precision_recall_fscore_support(y_val, predictions)[3]
      local_acc = np.diag(local_cm)/local_support
      kf_per_class_results.append(local_acc)

      # save results
      kf_results.to_csv("model/model_results.csv")

joblib.dump(best_classifier, Path(BASE_DIR).joinpath("model/mmama_model.joblib"))

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
elapsed / 60, elapsed))

class BasePredictor(object):
    def predict(self, X):
        raise NotImplementedError
    
class Predictor(BasePredictor):
    def __init__(self):
        self.model = best_classifier
        self.scaler = scaler

    def predict(self, X):
        # prediction
        data_to_pass = np.asarray(X)

        pred_data = self.scaler.transform(X = data_to_pass)

        result = self.model.predict(pred_data)

        # unpack results
        prediction = []

        for i in result:
            if i == 0:
                prediction.append("NoRisk")
            else:
                prediction.append("Risk")

        GH_prediction = np.asarray(prediction)

        return GH_prediction

joblib.dump(Predictor, "model/mmama_predictor.sav")
joblib.dump(Predictor, "model/mmama_predictor.joblib")


# define a convenient plotting function (confusion matrix)
def plot_confusion_matrix(
    cm,
    classes,
    normalise = True,
    text = False,
    title = 'Confusion matrix',
    xrotation = 0,
    yrotation = 0,
    cmap = plt.cm.Blues,
    printout = False
    ):
    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if printout:
            print("Normalized confusion matrix")

    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)

    plt.figure(figsize=(6, 4))
    plt.imshow(
        cm,
        interpolation = 'nearest',
        cmap = cmap,
        vmin = 0.2,
        vmax = 1.0
        )

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_ylim(len(classes)-0.5, -0.5)
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')
    
    fig1 = plt.gcf()
    fig1.savefig('figures/model_accuracy.png')
    #plt.close(fig1)
    


# plot averaged confusion matrix for training
averaged_CM = confusion_matrix(save_true, save_predicted)
classes = np.unique(np.sort(y_val))
plot_confusion_matrix(averaged_CM, classes, title="Confusion Matrix 1")

# features importance
model_results = pd. read_csv("model/model_results.csv")
features = pd.DataFrame(ast.literal_eval(model_results["Feature importances"][0]))

# iterate to get results from each validation
for record in model_results["Feature importances"][1:]:
    feature = pd.DataFrame(ast.literal_eval(record))
    features = pd.concat(
                [features, feature],
                axis=1,
                ignore_index=True
                )
    
# generate mean
features["mean"] = features.mean(axis=1)
features["sem"] = features.sem(axis=1)
features.sort_values(by="mean", inplace=True)

sns.set_theme(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.clf() # clear previous figure
fig2 = features["mean"].plot(
                                figsize = (6, 8),
                                kind = "barh",
                                # orientation = 'vertical',
                                legend = False,
                                xerr = features["sem"],
                                ecolor = 'k'
                            )
# plt.rcParams['ytick.labelsize'] = "small"
plt.title("Feature Importance", pad=20)
plt.tight_layout()
#plt.xlabel("Feature Importance", weight = 'bold')
sns.despine()
fig2 = plt.gcf()
fig2.savefig('figures/features_importance.png')

## Explain the modal prediciton using SHARP
# explainer = shap.Explainer(best_classifier)
# shap_values = explainer(X_train)

# feature_names = X.columns.tolist()
# shap_values.feature_names = feature_names

# # Adjust plot size
# plt.figure(figsize=(10, 6))  # Width = 10, Height = 5

# shap.plots.waterfall(shap_values[0], show=False)
# plt.tight_layout()

# fig3 = plt.gcf()
# fig3.savefig('figures/features_explainer.png')
# plt.close(fig3)

# ## summarize the effect of all figures
# plt.tight_layout()
# shap.plots.beeswarm(shap_values, show=False)
# fig4 = plt.gcf()
# fig4.savefig('figures/features_summary.png')
# plt.close(fig4)

# ensure best_classifier is fitted and X_train is a DataFrame
# Use TreeExplainer for XGBoost / tree models (fast and compatible)
try:
    explainer = shap.TreeExplainer(best_classifier)
    shap_values = explainer.shap_values(X_train)  # may be array or list (multiclass)
    # Normalize shap_values for plotting: pick class 0/1 for multiclass or use array directly
    if isinstance(shap_values, list):
        # choose class index to explain (e.g., class 1). adjust if you want another class.
        class_idx = 1 if len(shap_values) > 1 else 0
        shap_for_plot = shap_values[class_idx]
    else:
        shap_for_plot = shap_values

    # Build Explanation object with feature names if shap version requires it
    try:
        expl = shap.Explanation(values=shap_for_plot, feature_names=X_train.columns.tolist())
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(expl[0], show=False)
        plt.tight_layout()
        plt.savefig('figures/features_explainer.png')
        plt.close()
    except Exception:
        # fallback if shap.Explanation not available
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_for_plot[0], show=False)
        plt.tight_layout()
        plt.savefig('figures/features_explainer.png')
        plt.close()

    # summary / beeswarm
    plt.figure(figsize=(10, 6))
    # if list, pass the array for chosen class; else pass shap_values directly
    shap.plots.beeswarm(shap_values if not isinstance(shap_values, list) else shap_values[class_idx], show=False)
    plt.tight_layout()
    plt.savefig('figures/features_summary.png')
    plt.close()

except Exception as e:
    # last-resort: use shap.Explainer with a callable (predict_proba) if TreeExplainer fails
    try:
        print("TreeExplainer failed, falling back to shap.Explainer with predict_proba:", e)
        explainer = shap.Explainer(best_classifier.predict_proba, X_train)
        shap_vals = explainer(X_train)
        plt.figure(figsize=(10,6))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.tight_layout()
        plt.savefig('figures/features_explainer.png')
        plt.close()

        plt.figure(figsize=(10,6))
        shap.plots.beeswarm(shap_vals, show=False)
        plt.tight_layout()
        plt.savefig('figures/features_summary.png')
        plt.close()
    except Exception as e2:
        print("SHAP explanation unavailable:", e2)

## Predict test (validation) data
X_test_scl = scaler.transform(X = X_test)
y_pred = best_classifier.predict(X_test_scl)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


## print auc score isaac
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score for the model: {auc_score}")

## Plotting ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#plt.figure()
plt.figure(figsize = (15, 15))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('figures/roc_curve1.png')
plt.clf()   # clear figure
#plt.show()

# Plotting confusion matrix unseen test data
cm_unseen = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm_unseen, classes = classes, title="Confusion Matrix 2")


# Classification Report
cr = classification_report(y_test, y_pred)
print('Classification report : {}')
print(cr)