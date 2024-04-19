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

##some basic data observation
#anc_head()
#anc_visits.info()
#anc_visits.shape
#anc_visits['blood_for_glucose'].describe()
#anc_visits['visit_number'].value_counts()

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

#separating categorical and numerical variables
categorical_attributes = ['ganc','protein_in_urine','glucose_in_urine','blood_group','syphilis']
numerical_attributes = [i for i in data4.columns if i not in categorical_attributes]



#for individual cross-section analysis : individual observation
data4_cat = data4[categorical_attributes]
data4_num  = data4[numerical_attributes]
#drop ganc and client_id, we don't need them pass this point
#data4_cat = data4_cat.drop(['ganc'], axis = 1)
data4_num = data4_num.drop(['client_id'], axis = 1)

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

# define outcome condictions based on detection cutpoints
# this is the same as data labeling
chronic_hypertension = (data5['systolic'] >= 140) & (data5['diastolic'] >= 90) & (data5['gest_age'] <= 20)
gestational_hypertension = (data5['systolic'] >= 140) & (data5['diastolic'] >= 90) & (data5['gest_age'] >= 20)
pre_eclampsia = (data5['systolic'] >= 140) & (data5['diastolic'] >= 90) & (data5['protein_in_urine'] == 1) & (data5['gest_age'] >= 20)



data5['label'] = 0
data5.loc[chronic_hypertension, 'label'] = 1
data5.loc[gestational_hypertension, 'label'] = 2
data5.loc[pre_eclampsia, 'label'] = 3

# print data labeling
print('Detect outcome conditions by cutpoints')
print('0:No-Risk, 1: Chronic Hypertension, 2: Gestational Hypertension, 3: Pre-eclampsia \n', data5['label'].value_counts())

# reduce the dataset then drop NA
data6 = data5[['systolic', 'diastolic', 'protein_in_urine', 'temperature', 'bmi', 'blood_for_glucose', \
               'syphilis', 'label']]
print('label count before dropping NA:', data6['label'].value_counts())

# index data
label_2_rows = data6[data6['label'] == 2]

# fill NA with median value
median_values = label_2_rows.drop('label', axis = 1).median()
for index, row in label_2_rows.iterrows():
    data6.loc[index] = data6.loc[index].fillna(median_values)

data6_no_NAs = data6.dropna() # this to be discussed, and presented for discussion
print('label count after dropping NA:', data6_no_NAs['label'].value_counts())

training_data = data6_no_NAs.loc[(data6_no_NAs['label'] == 0) | (data6_no_NAs['label'] == 2)]
print('training data', training_data['label'].value_counts())


# Rule of thumb balance the instances, when one class is overepresented
# Here we will undersample the majority class to match the minority class
# The aim is to avoid bias

# define X (matrix of features) and y (list of labels)

X = training_data.iloc[:, :-1] # select all columns except the last one
y = training_data["label"]

# Balance the training data
# return x-resampled and y-resampled
rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)
print('The number of samples after resampling:', Counter(y_res))

# Concatinate resampled data for explorations by plots

temp_df = pd.concat([X_res,y_res],axis = 1).reset_index(drop = True)

#drawing figures
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context = "paper", style = "white", palette = "deep", font_scale = 2.0, color_codes = True, rc = ({'font.family': 'Dejavu Sans'}))
plt.figure(figsize = (15, 15))
fig, axs = plt.subplots(1, 3)

sns.violinplot(x = "label",y = "systolic",hue = 'label',data = temp_df,palette = 'colorblind',ax = axs[0], legend = False)
sns.violinplot(x = "label",y = "diastolic",hue = 'label',data = temp_df,palette = 'colorblind',ax = axs[1], legend = False)
sns.violinplot(x = "label",y = "temperature",hue = 'label',data = temp_df,palette = 'colorblind',ax = axs[2],legend = False)
fig.tight_layout(pad = .5)
fig.savefig('figures/violine.png')

plt.rcParams["figure.figsize"] = [15,15]
sns.pairplot(temp_df, hue = "label", height = 3, palette = 'colorblind').savefig('figures/scatterplots.png')


plt.rcParams["figure.figsize"] = [15,10]
sns.heatmap(temp_df.corr(method = 'pearson'),annot = True,vmin = -1,vmax = 1).get_figure().savefig('figures/heatmap1.png')

# drop protein in urine as it has no influence in the model
sns.heatmap(temp_df.drop(['protein_in_urine'], axis = 1).corr(method = 'pearson'),annot = True,vmin = -1,vmax = 1).get_figure().savefig('figures/heatmap2.png')

# Split training and test data
# We wont drop glucose in urine as for now, until we have discussed and agreed
from sklearn.model_selection import (GridSearchCV,train_test_split,cross_val_score,RandomizedSearchCV,KFold)
X_res_temp = np.asarray(X_res)
y_res_temp = np.asarray(y_res)

# our y_res has [0, 2], would like to change it [0, 1]
y_res_temp = np.where(y_res_temp == 2, 1, y_res_temp)
y_res_temp

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
models.append(('LR', LogisticRegression(multi_class = 'ovr',max_iter = 2000,random_state = seed)))
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

# XGB will expect the unique values of `y`. [0 1], but in this our currrent y_res got [0 2]
# replace 2 with 1
y_res_2 = y_res.replace(2, 1)

print("A3")
# fit algorithm
pipesfs = pipe.fit(X_res, y_res_2)
print("A4")
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
pipe.fit(X_res, y_res_2)

# Get the selected feature indices
selected_features_indices = pipe.named_steps['rfecv'].support_

# Get the names of the final features
final_feature_names = X_res.columns[selected_features_indices]

# Print the final selected features
print("Selected Features:", final_feature_names)

# Evaluate models to get the best perfoming model

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model,scl_features,y_train,cv = kf,scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = 'Cross validation score for {0}: {1:.2%}'.format(name,cv_results.mean(),cv_results.std())
    print(msg)


# Plot results for algorithm comparison

# transform the vectors into pandas dataframe
results_df = pd.DataFrame(results,columns = (0, 1, 2, 3, 4)).T # columns should correspond to the number of folds, k = 5

results_df.columns = names
results_df = pd.melt(results_df) # melt data frame into a long format.
results_df.rename(columns = {'variable':'Model', 'value':'Accuracy'},inplace = True)


# Plotting the algorithm selection

plt.figure(figsize = (6, 4))

sns.boxplot(x = 'Model',y = 'Accuracy',data = results_df,hue = 'Model',legend = False,palette = 'colorblind')
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .1, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ")
plt.tight_layout()
plt.savefig('figures/algorithm_comparison.png')


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

      print('The shape of X train set : {}'.format(X_train_set.shape))
      print('The shape of y train set : {}'.format(y_train_set.shape))
      print('The shape of X validation : {}'.format(X_val_set.shape))
      print('The shape of y validation : {}'.format(y_val.shape))

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
      for mean, stdev, param in zip(means, stds, params):
          print("%.2f (%.2f) with: %r" % (mean, stdev, param))

      # print best parameter settings
      print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                  rsCV_result.best_params_))

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

joblib.dump(best_classifier, Path(BASE_DIR).joinpath("model/mmama_classifier.joblib"))

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
                prediction.append("GH_Negative")
            else:
                prediction.append("GH_Positive")

        GH_prediction = np.asarray(prediction)

        return GH_prediction

joblib.dump(Predictor, "model/mmama_predictor.sav")
joblib.dump(Predictor, "model/mmama_predictor.joblib")