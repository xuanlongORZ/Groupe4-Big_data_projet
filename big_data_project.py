import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from tqdm import tqdm

# load data
data = pd.read_csv('train.csv')
(nb_sample,nb_feature) = data.shape
print("Training set loading - Done!")

# delete samples with 14 and 15 missing featrues
nb_delete_1415 = 0
for i in range(nb_sample):
    if i < (nb_sample-nb_delete_1415):
        nb_missing = data.iloc[i,:].isnull().sum()
        if nb_missing >= 13:
            data.drop(index=[i],inplace=True)
            nb_delete_1415+=1
    else:
        break
data.reset_index(drop = True,inplace = True)

# delete directly the features that have more than 70% missing values
del data['Medical_History_10'],data['Medical_History_32'],data['Medical_History_24'],data['InsuredInfo_8'],data['Medical_History_15'],data['InsuredInfo_9'],data['Family_Hist_5']

# onehot encoding: InsuredInfo_7; label encoding: Product_Info_2
data = data.join(pd.get_dummies(data['InsuredInfo_7']))
del data['InsuredInfo_7']
le = LabelEncoder().fit(data['Product_Info_2'])
data['Product_Info_2'] = le.transform(data['Product_Info_2'])

# Fill missing values by mean for other features
missing_features = pd.DataFrame(data.dtypes,columns=['dtypes'])
missing_features = missing_features.reset_index()
missing_features['Name'] = missing_features['index']
del missing_features['index']
missing_features.reset_index(drop=True,inplace=True)
missing_features = missing_features[data.isnull().sum().values!=0]

# as they are all float data, we can then fill them by mean directly
data.fillna(data.mean()[missing_features.Name],inplace=True)

# See variance again and delete top5 features with the lowest variance
tabel_var = data[data.columns].var().sort_values(ascending=True)
tabel_var = tabel_var.head(5)
tabel_var.reset_index()
for column in tabel_var.index.tolist():
    del data[column]

# See correlation again to delete top5 features with the lowest correlation to the targe
X_temp = data
X_temp = X_temp.corr()
tabel_corr = abs(X_temp[['Response']]).sort_values(by=['Response'],ascending=True)
tabel_corr = tabel_corr.head(5)
tabel_corr.reset_index()
for column in tabel_corr.index.tolist():
    del data[column]
del X_temp

# delete the outliers
data.drop(data[data['Medical_History_18']==3].index,axis=0,inplace=True)
data.drop(data[data['Medical_History_28']==3].index,axis=0,inplace=True)
data.drop(data[data['Medical_History_39']==2].index,axis=0,inplace=True)
data.drop(data[data['Medical_History_13']==2].index,axis=0,inplace=True)
data.drop(data[data['Medical_History_40']==2].index,axis=0,inplace=True)
data.drop(data[data['Medical_History_30']==1].index,axis=0,inplace=True)
data.drop(data[data['InsuredInfo_3']==9].index,axis=0,inplace=True)
data.reset_index(drop=True,inplace=True)

# seperate label and features
y_data = data['Response']
del data['Response']

# important features
important_features = ['BMI', 'Product_Info_4', 'Medical_History_4', 'Ins_Age', 'Wt', 'Medical_History_23', 'Medical_Keyword_3', 'Medical_Keyword_15', 'Product_Info_2', 'Employment_Info_1', 'Family_Hist_3', 'Medical_History_1', 'Family_Hist_4', 'Employment_Info_6', 'Medical_History_2', 'Family_Hist_2', 'Employment_Info_2', 'Medical_History_30', 'Female', 'InsuredInfo_3', 'Medical_History_40', 'Medical_History_13', 'Medical_History_39', 'Medical_History_28', 'Insurance_History_2', 'Family_Hist_1', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_2', 'Medical_History_18', 'InsuredInfo_1', 'Product_Info_3', 'Insurance_History_8', 'Medical_History_33', 'Medical_Keyword_23']
data = data[important_features]

print("Training set data processing - Done!")


# seperate training set and validation set
X_train, X_validation, y_train, y_validation = train_test_split(data, y_data,
                                                                test_size=0.15, stratify=y_data,random_state=0)

# xgboost model
X_train.reset_index(drop=True,inplace=True)
X_validation.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_validation.reset_index(drop=True,inplace=True)
folds = 4
seed = 0
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
for tr_idx, val_idx in tqdm(kf.split(X_train,y_train)):
    def fit_classifier(tr_idx, val_idx):
        tr_x, tr_y = X_train[list(X_train)].iloc[tr_idx], y_train[tr_idx]
        vl_x, vl_y = X_train[list(X_train)].iloc[val_idx], y_train[val_idx]
        print({'train size':len(tr_x), 'eval size':len(vl_x)})

        clf = xgb.XGBClassifier(n_estimators=1000,
                                learning_rate=0.01,
                                feature_fraction=0.8,
                                max_depth = 10,
                                subsample=0.25,
                                subsample_freq=1,
                                lambda_l2=0.8,
                                num_leaves=30,
                                metric='mse')
        clf.fit(tr_x, tr_y,
                eval_set=[(tr_x, tr_y),(vl_x, vl_y)],
                early_stopping_rounds=150,
                verbose=250)
        return clf
    clf = fit_classifier(tr_idx, val_idx)
    models.append(clf)
print("4 Xgboost models training - Done!")

# same operations on test set
X_test = pd.read_csv('predict.csv')
print("Test set loading - Done!")

del X_test['Medical_History_10'],X_test['Medical_History_32'],X_test['Medical_History_24'],X_test['InsuredInfo_8'],X_test['Medical_History_15'],X_test['InsuredInfo_9'],X_test['Family_Hist_5']
X_test = X_test.join(pd.get_dummies(X_test['InsuredInfo_7']))
del X_test['InsuredInfo_7']
le = LabelEncoder().fit(X_test['Product_Info_2'])
X_test['Product_Info_2'] = le.transform(X_test['Product_Info_2'])
missing_features = pd.DataFrame(X_test.dtypes,columns=['dtypes'])
missing_features = missing_features.reset_index()
missing_features['Name'] = missing_features['index']
del missing_features['index']
missing_features.reset_index(drop=True,inplace=True)
missing_features = missing_features[X_test.isnull().sum().values!=0]
X_test.fillna(X_test.mean()[missing_features.Name],inplace=True)
X_test = X_test[important_features]
X_test.reset_index(drop=True,inplace=True)
print("Test set data processing - Done!")

# prediction only use 3 xgboost models to make a hard voting
res_preds = []
for i in range(3):
  res_preds.append(models[i+1].predict(X_test))

res = []
for i in range(len(X_test)):
  res_temp = []
  for j in range(3):
    res_temp.append(res_preds[j][i])
  res.append(np.argmax(np.bincount(res_temp)))

pred = pd.DataFrame(res)
pred.rename(columns={0:"Response"},inplace=True)
print("Prediction - done!")

pred.to_csv("submission.csv",index=False)
