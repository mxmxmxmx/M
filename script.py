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
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


start_time = time.time()
#NUM_BRANDS = 2500
NAME_MIN_DF = 10
#MAX_FEAT_DESCP = 50000
#NUM_BRANDS = 4000
MAX_FEAT_DESCP = 100000
NUM_BRANDS = 4000

folds = 3
print("Reading in Data")

df_train = pd.read_csv('train.tsv', engine='python', sep='\t')
df_test = pd.read_csv('test.tsv', engine='python', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

df["category_name"] = df["category_name"].fillna("Other").astype("category")
df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
    
del df_train
gc.collect()

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
    "item_condition_id", "shipping"]], sparse = True).values)

print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(df["item_description"])

df["brand_name"] = df["brand_name"].fillna("unknown")
   
trainerror_nbrofDescps = np.zeros(len(NUM_BRANDS))
valerror_nbrofDescps = np.zeros(len(NUM_BRANDS)) 

for index,NUM_BRAND in enumerate(NUM_BRANDS):    
    pop_brands = df["brand_name"].value_counts().index[:NUM_BRAND]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"
    df["brand_name"] = df["brand_name"].astype("category")
    
    print("Brand encoders")
    vect_brand = LabelBinarizer(sparse_output=True)
    X_brand = vect_brand.fit_transform(df["brand_name"])
    
    
    X = scipy.sparse.hstack((X_dummies, 
                             X_descp,
                             X_brand,
                             X_category,
                             X_name)).tocsr()
    print([X_dummies.shape, X_category.shape, 
           X_name.shape, X_descp.shape, X_brand.shape])
    
    X_train = X[:nrow_train]
    X_test = X[nrow_train:]
#    Xtrain, Xval, ytrain, yval = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)
#
#    print("Fitting Model: number %d " %index)
# 
#    model = Ridge(alpha = 5, solver = "sag", fit_intercept=False)
#    model.fit(Xtrain, ytrain)
#    ridge_train = model.predict(Xtrain)
#    ridge_train_score = np.sqrt(mean_squared_error(ytrain, ridge_train))
#
#    ridge_val = model.predict(Xval)  
#    ridge_cv_score = np.sqrt(mean_squared_error(yval, ridge_val))
#    ridge_currentPredict = model.predict(X_test)
#
#    print('\n Average Ridge RMSLE on validation:\t%.6f' % ridge_cv_score)
#    print('\n Average Ridge RMSLE on training:\t%.6f' % ridge_train_score)
#    
#    trainerror_nbrofDescps[index] = ridge_train_score
#    valerror_nbrofDescps[index] = ridge_cv_score
#    
    model = Ridge(alpha = 5, solver = "sag", fit_intercept=False)
    plot_learning_curve(model, 'test', X_train, y_train, cv=3, n_jobs=4)


#blend_test = ridge_pred / folds
#df_test["price"] = np.expm1(blend_test)
#df_test[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)

plt.plot(NUM_BRANDS, trainerror_nbrofDescps)
plt.plot(NUM_BRANDS, valerror_nbrofDescps)
plt.legend(['ridge_train_score', 'valerror_nbrofDescps'])