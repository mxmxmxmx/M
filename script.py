"""
Ridge Regression on TfIDF of text features and One-Hot-Encoded Categoricals
"""

import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gc
import time

start_time = time.time()
#NUM_BRANDS = 2500
NAME_MIN_DF = 10
#MAX_FEAT_DESCP = 50000
NUM_BRANDS = 4000
MAX_FEAT_DESCP = 100000

folds = 3
print("Reading in Data")

df_train = pd.read_csv('train.tsv', engine='python', sep='\t')
df_test = pd.read_csv('test.tsv', engine='python', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

del df_train
gc.collect()

print(df.memory_usage(deep = True))

df["category_name"] = df["category_name"].fillna("Other").astype("category")
df["brand_name"] = df["brand_name"].fillna("unknown")

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
df["brand_name"] = df["brand_name"].astype("category")

print(df.memory_usage(deep = True))

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
    "item_condition_id", "shipping"]], sparse = True).values)

trainerror_nbrofDescps = np.zeros(len(MAX_FEAT_DESCPS))
valerror_nbrofDescps = np.zeros(len(MAX_FEAT_DESCPS))

for index,MAX_FEAT_DESCP in enumerate(MAX_FEAT_DESCPS):
        
    print("Descp encoders")
    count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                                  ngram_range = (1,3),
                                  stop_words = "english")
    X_descp = count_descp.fit_transform(df["item_description"])
    
    X = scipy.sparse.hstack((X_dummies, 
                             X_descp,
                             X_brand,
                             X_category,
                             X_name)).tocsr()
    
    print([X_dummies.shape, X_category.shape, 
           X_name.shape, X_descp.shape, X_brand.shape])
    
    X_train = X[:nrow_train]
    X_test = X[nrow_train:]
    
    model = Ridge(solver = "lsqr", fit_intercept=False)
    
    print("Fitting Model")
    ridge_avpred = np.zeros(X_train.shape[0])
    
    ridge_cv_sum = 0
    ridge_train_sum = 0
    
    kf = KFold(n_splits=folds, random_state=1001)
    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        Xtrain, Xval = X_train[train_index], X_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]
        
        model = Ridge(solver = "sag", fit_intercept=False)
        model.fit(Xtrain, ytrain)
        ridge_train = model.predict(Xtrain)
        ridge_train_RMSLE = np.sqrt(mean_squared_error(ytrain, ridge_train))
    
        ridge_val = model.predict(Xval)  
        ridge_RMSLE = np.sqrt(mean_squared_error(yval, ridge_val))
        ridge_currentPredict = model.predict(X_test)
        
        del Xtrain, Xval
        gc.collect()
    
        ridge_avpred[val_index] = ridge_val
    
        if i == 0:
            ridge_pred = ridge_currentPredict
        else:
            ridge_pred = ridge_pred + ridge_currentPredict
    
        ridge_cv_sum = ridge_cv_sum + ridge_RMSLE
        ridge_train_sum = ridge_train_sum + ridge_train_RMSLE
    ridge_cv_score = (ridge_cv_sum / folds)
    ridge_train_score = ridge_train_sum/folds
    ridge_oof_RMSLE = np.sqrt(mean_squared_error(y_train, ridge_avpred))
    
    print('\n Average Ridge RMSLE on validation:\t%.6f' % ridge_cv_score)
    print(' Out-of-fold Ridge RMSLE:\t%.6f' % ridge_oof_RMSLE)
    print('\n Average Ridge RMSLE on training:\t%.6f' % ridge_train_score)
    
    trainerror_nbrofDescps[index] = ridge_train_score
    valerror_nbrofDescps[index] = ridge_oof_RMSLE
blend_test = ridge_pred / folds
df_test["price"] = np.expm1(blend_test)
df_test[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)

plt.plot(MAX_FEAT_DESCPS, trainerror_nbrofDescps)
plt.plot(MAX_FEAT_DESCPS, valerror_nbrofDescps)
plt.legend(['ridge_train_score', 'valerror_nbrofDescps'])