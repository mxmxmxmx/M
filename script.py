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
alphas = np.logspace(0, 2, 10)

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
   
trainerror_nbrofDescps = np.zeros(len(alphas))
valerror_nbrofDescps = np.zeros(len(alphas)) 
Xtrain, Xval, ytrain, yval = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

for index,alp in enumerate(alphas):    
    print("Fitting Model: number %d " %index)
 
    model = Ridge(alpha = alp, solver = "sag", fit_intercept=False)
    model.fit(Xtrain, ytrain)
    ridge_train = model.predict(Xtrain)
    ridge_train_score = np.sqrt(mean_squared_error(ytrain, ridge_train))

    ridge_val = model.predict(Xval)  
    ridge_cv_score = np.sqrt(mean_squared_error(yval, ridge_val))
    ridge_currentPredict = model.predict(X_test)

    print('\n Average Ridge RMSLE on validation:\t%.6f' % ridge_cv_score)
    print('\n Average Ridge RMSLE on training:\t%.6f' % ridge_train_score)
    
    trainerror_nbrofDescps[index] = ridge_train_score
    valerror_nbrofDescps[index] = ridge_cv_score
    
#blend_test = ridge_pred / folds
#df_test["price"] = np.expm1(blend_test)
#df_test[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)

plt.plot(alphas, trainerror_nbrofDescps)
plt.plot(alphas, valerror_nbrofDescps)
plt.legend(['ridge_train_score', 'valerror_nbrofDescps'])