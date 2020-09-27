# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:28:05 2020

@author: Viola
"""
import pandas as pd
import numpy as np

import data_cleaning as dc
import feature_engineering as fe

import matplotlib.pyplot as plt
from numpy import hstack

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import precision_recall_curve

import imblearn

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier


from collections import Counter

from copy import deepcopy

#input data (data cleaning and feature combing are done)

train_data=pd.read_csv("train_data_feature.csv")
test_data=pd.read_csv("test_data_feature.csv")

'''
Part one Create train&test dataset (Use oversampling strategy to deal with imbalanced classification)

'''
train_y=train_data["TARGET"]
train_x_imbalanced=train_data.drop(columns=["SK_ID_CURR","TARGET"])

test_x=test_data.drop(columns=["SK_ID_CURR"])

train_x_imbalanced.replace([np.inf,-np.inf],np.nan,inplace=True)
test_x.replace([np.inf,-np.inf],np.nan,inplace=True)

#train&test align

train_x_imbalanced, test_x = train_x_imbalanced.align(test_x, join = 'inner', axis = 1)

#rename columns to handle error

column_name=[name.replace("/","_").replace("+","_").replace("(","_").replace(")","_") for name in train_x_imbalanced.columns]

train_x_imbalanced.columns=column_name
test_x.columns=column_name

#label encoding
def trans(data):
   #label encoding & fillna
    label_encoder = preprocessing.LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
                    data[col].fillna("nan",inplace=True)
                    data[col]=label_encoder.fit_transform(data[col])
                    
        else:
            data[col].fillna(data[col].mean(),inplace=True)
    return data

train_x_imbalanced=trans(train_x_imbalanced)
test_x=trans(test_x)

# define oversampling strategy
x_train, x_valid,y_train, y_valid = train_test_split(train_x_imbalanced, train_y, test_size = 0.2, random_state = 405)
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy="minority")
train_x, train_y = oversample.fit_resample(x_train, y_train)


'''
Part two Model Application, Hyperparameter tuning and Model Evaluation
'''

#run performance matrix report from out-of-sample training dataset
def Performance_Metrics(model,model_name,x_valid,y_valid):

    y_pred = model.predict(x_valid)
    pre_metrics={"accuracy":metrics.accuracy_score(y_valid, y_pred),
             "precision":metrics.precision_score(y_valid, y_pred),
             "recall":metrics.recall_score(y_valid, y_pred),
             "F1 score":metrics.f1_score(y_valid, y_pred)}
    print(pre_metrics)
    print( metrics.confusion_matrix(y_valid, y_pred))

    # plot ROC curve and calculate AUC 
    y_pred_prob = model.predict_proba(x_valid)
    # keep probabilities for the positive class only
    y_pred_prob = y_pred_prob[:, 1]
    # calculate the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_prob)
    # Calculate AUC
    AUC=metrics.roc_auc_score(y_valid, y_pred_prob)
    print('AUC:', AUC)
   

    plt.plot([0, 1], [0, 1], ls = '--')
    plt.plot(fpr, tpr,color="c",label="{} AUC {}".format(model_name,np.round(AUC,2)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title('{} ROC Curve'.format(model_name))
    plt.show()
    
    #plot precision & recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_valid, y_pred_prob)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "c-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title('{} Precision-Recall Tradeoff'.format(model_name))
    plt.ylim([0, 1])
    
    plt.figure()
    plt.plot(recalls[:-1],precisions[:-1],"c-")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title('{} Precision-Recall'.format(model_name))
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    
    PR_AUC=metrics.auc(recalls, precisions)
    print("PR_AUC",PR_AUC)

# run performance comparison among different model
def performance_comparison(models,model_names,x_valid,y_valid):
    plt.figure()
    plt.plot([0, 1], [0, 1], ls = '--')
    plt.xlabel('FPR')
    plt.ylabel('TPR') 
    n=1/len(models)
    i=1
    for model,model_name in zip(models,model_names):
        # plot ROC curve and calculate AUC 
        y_pred_prob = model.predict_proba(x_valid)
        # keep probabilities for the positive class only
        y_pred_prob = y_pred_prob[:, 1]
        # calculate the ROC curve
        AUC=metrics.roc_auc_score(y_valid, y_pred_prob)
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_prob)
        plt.plot(fpr, tpr,ls = '-',color="c",label="{} AUC {}".format(model_name,np.round(AUC,2)),alpha=n*i)
        plt.legend()
        i=i+1

# 1. Navie Bayes
#1.1
NB_model = GaussianNB()
NB_model.fit(train_x, train_y)
Performance_Metrics(NB_model,"Naive Bayes",x_valid, y_valid)

# apply ensembles
#1.2.1 bagging
bag_NB=BaggingClassifier(NB_model,n_estimators=100,max_samples=100, bootstrap=True)
bag_NB.fit(train_x, train_y)
Performance_Metrics(bag_NB,"Naive Bayes_bagging",x_valid, y_valid)
#1.2.2 boosting
ada_NB = AdaBoostClassifier(NB_model, n_estimators=50,learning_rate=0.5)
ada_NB.fit(train_x, train_y)
Performance_Metrics(ada_NB,"Naive Bayes_adaboost",x_valid, y_valid)

#Performance comparision
performance_comparison([NB_model,bag_NB,ada_NB],["Naive Bayes","Naive Bayes_bagging","Naive Bayes_adaboost"],x_valid,y_valid)


# 2. logtistic regression with grid search
logistic_model = LogisticRegression(solver = 'lbfgs')
#2.1.1 without PCA
standard_train_x = StandardScaler().fit_transform(train_x)
standard_valid_x = StandardScaler().fit_transform(x_valid)

grid={"C":np.logspace(-3,3,7)}
logistic_model_cv=GridSearchCV(logistic_model,grid,scoring="roc_auc")

logistic_model_cv.fit(standard_train_x, train_y)
Performance_Metrics(logistic_model_cv,"Logistic Regression",standard_valid_x, y_valid)
#get best parameter
C_logistic_best=logistic_model_cv.best_params_

#2.1.2 with PCA
pca_train_x=fe.PCA_feature(standard_train_x )
pca_valid_x=fe.PCA_feature(x_valid)

fe.PCA_plot(standard_train_x)

# align train and test components
components=min(pca_train_x.shape[1],pca_valid_x.shape[1])
pca_train_x=standard_train_x[:,:components]
pca_valid_x=pca_valid_x[:,:components]
logistic_model_cv.fit(pca_train_x, train_y)
Performance_Metrics(logistic_model_cv,"Logistic Regression with PCA",pca_valid_x, y_valid)

# apply ensembles without PCA
logistic_model_cv.fit(standard_train_x, train_y)
#2.2.1 bagging

bag_log=BaggingClassifier(logistic_model_cv,n_estimators=100,max_samples=100, bootstrap=True)
bag_log.fit(standard_train_x, train_y)
Performance_Metrics(bag_log,"Logistic Regression_bagging",standard_valid_x, y_valid)

#2.2.2 boosting
logistic_model = LogisticRegression(solver = 'lbfgs',C=100)
ada_log=AdaBoostClassifier(logistic_model, n_estimators=100,learning_rate=0.5)
ada_log.fit(standard_train_x, train_y)
Performance_Metrics(ada_log,"Logistic Regression_adaboost",standard_valid_x, y_valid)

#compare
performance_comparison([logistic_model_cv,bag_log,ada_log],["Logistic Regression","Logistic Regression_bagging","Logistic Regression_adaboost"],standard_valid_x,y_valid)


#3. lightgmb

lightgbm_model = lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.05,n_jobs=-1,verbose = -1)


# Initialize an empty array to hold feature importances
feature_importance_value = np.zeros(train_x.shape[1])

for i in range(10):
    
    x_train_1, x_valid_1,y_train_1, y_valid_1 = train_test_split(train_x, train_y, test_size = 0.2, random_state = i)
        
    lightgbm_model.fit(x_train_1,y_train_1.astype(int),eval_metric='auc',
             eval_set = [(x_valid_1, y_valid_1.astype(int))],
             early_stopping_rounds=100,verbose = -1)
    feature_importance_value += lightgbm_model .feature_importances_

feature_importance_value=feature_importance_value/10
df_feature_importance_lgb = pd.DataFrame({'feature':train_x.columns,
                           'feature_importance':feature_importance_value}).sort_values(by='feature_importance',ascending=False).reset_index(drop=True)

Performance_Metrics(lightgbm_model,"LightGBM",x_valid,y_valid)

#plot feature importance

def plot_feature_importance(df_feature_importance):
    df_feature_importance['norm_importance'] = df_feature_importance['feature_importance']/df_feature_importance['feature_importance'].sum()
    df_feature_importance['cum_importance'] = np.cumsum(df_feature_importance['norm_importance'])
    
    plt.figure(figsize=(16,10))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df_feature_importance.index[:15]))), df_feature_importance['norm_importance'].head(15),align = 'center', edgecolor = 'c')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df_feature_importance.index[:15]))))
    ax.set_yticklabels(df_feature_importance['feature'].head(15))
    plt.xlabel('Normalized Importance')
    plt.title('Feature importance')
    
    
    plt.figure(figsize=(16,10))
    plt.title('Feature Cumulative Importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Feature Cumulative Importance')
    plt.plot(list(range(1, len(df_feature_importance)+1)),df_feature_importance['cum_importance'], 'c-')

#plot
plot_feature_importance(df_feature_importance_lgb)
# select features using feature importance

zero_importance=df_feature_importance_lgb[df_feature_importance_lgb.feature_importance==0]["feature"].values

train_x_lgb=train_x.drop(columns=zero_importance)
valid_x_lgb=x_valid.drop(columns=zero_importance)

train_x_lgb, valid_x_lgb = train_x_lgb.align(valid_x_lgb, join = 'inner', axis = 1)

# create new lightgbm model with selected features without hyperparameter tuning
lightgbm_model_selected = lgb.LGBMClassifier(random_state=1)
lightgbm_model_selected.fit(train_x_lgb,train_y)
Performance_Metrics(lightgbm_model_selected,"LightGBM",valid_x_lgb, y_valid)

'''
Hyperparameter tuning--Bayesian Optimization for LightGBM
'''

def bayes_parameter_lgb(x, y, init_round=5, opt_round=5, n_folds=5,
                        random_seed=6, n_estimators=100, learning_rate=0.05):
    # prepare data
    train_data = lgb.Dataset(data=x, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, 
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 
                  'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed,
                           stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    
    return lgbBO.max

train_x_lgb.fillna(0,inplace=True)
valid_x_lgb.fillna(0,inplace=True)

lgb_param=bayes_parameter_lgb(train_x_lgb,train_y)
best_param=lgb_param["params"]

lightgbm_model_ht = lgb.LGBMClassifier(boosting_type='gbdt',
                                       objective = 'binary',
                                       metric = 'auc',
                                       bagging_fraction= best_param['bagging_fraction'],
                                       feature_fraction= best_param['feature_fraction'],
                                       lambda_l1=best_param['lambda_l1'],
                                       lambda_l2= best_param['lambda_l2'],
                                       max_depth= best_param['max_depth'].astype(int),
                                       min_child_weight= best_param['min_child_weight'],
                                       min_split_gain= best_param['min_split_gain'],
                                       num_leaves=best_param['num_leaves'].astype(int),
                                       random_state=2)



lightgbm_model_ht.fit(train_x_lgb,train_y)
Performance_Metrics(lightgbm_model_ht,"LightGBM_tuned",valid_x_lgb, y_valid)

#bagging

bag_gbm=BaggingClassifier(lightgbm_model_ht,n_estimators=100,bootstrap=True)
bag_gbm.fit(train_x_lgb, train_y)
Performance_Metrics(bag_gbm,"LightGBM_bagging",valid_x_lgb, y_valid)

#boosting

ada_gbm=AdaBoostClassifier(logistic_model, n_estimators=100,learning_rate=0.5)
ada_gbm.fit(train_x_lgb, train_y)
Performance_Metrics(ada_gbm,"LightGBM_adaboost",valid_x_lgb, y_valid)

#compare
performance_comparison([lightgbm_model_selected,lightgbm_model_ht,bag_gbm],["LightGBM","LightGBM_tuned","LightGBM_bagging"],valid_x_lgb,y_valid)


#4. random forest
#4.1
rf = RandomForestClassifier()
rf.fit(train_x, train_y)
Performance_Metrics(rf,"Random Forest",x_valid, y_valid)


feature_importance_value_rf = rf .feature_importances_
df_feature_importance_rf = pd.DataFrame({'feature':train_x.columns,
                           'feature_importance':feature_importance_value_rf}).sort_values(by='feature_importance',ascending=False).reset_index(drop=True)

plot_feature_importance(df_feature_importance_rf)


def bayes_parameter_rf(x,y,init_round=5, opt_round=5, n_folds=2):
    def rf_eval(n_estimators, max_depth, min_samples_split,min_samples_leaf):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)), 
                   min_samples_leaf=int(max(min_samples_leaf,1)),
                   random_state=3,   
                   class_weight="balanced"),  
               X=x, 
               y=y, 
               cv=n_folds,
               scoring="recall",
                ).mean()

    parameters = {"n_estimators": (0, 50),
                  "max_depth": (1, 10),
                  "min_samples_split": (2, 10),
                  'min_samples_leaf':(2,10)}


    rfBO = BayesianOptimization(rf_eval, parameters)
    rfBO.maximize(init_points=init_round,n_iter=opt_round)

    return rfBO.max


rf_param=bayes_parameter_rf(train_x,train_y)
rf_param_best=rf_param["params"]


rf_ht = RandomForestClassifier(n_estimators= rf_param_best['n_estimators'].astype(int),
                               max_depth= rf_param_best['max_depth'].astype(int),
                               min_samples_split=rf_param_best['min_samples_split'].astype(int),
                               min_samples_leaf= rf_param_best['min_samples_leaf'].astype(int),
                               random_state=4)



rf_ht.fit(train_x,train_y)
Performance_Metrics(rf_ht,"Random Forest_tuned",x_valid, y_valid)



'''
Model Combination (Voting & Stacking)
'''

#Voting
voting_clf = VotingClassifier(
estimators=[("log",logistic_model),("rf",rf_ht)],
 voting='hard'
 )
#voting_clf.fit(train_x,train_y)
#Performance_Metrics(voting_clf,"Voting Classifer",x_valid, y_valid)

#create out-of-sampe prediction

data_x, data_y, y_pred,y_pred_2 = pd.DataFrame(), list(), pd.DataFrame(),pd.DataFrame()
kfold = KFold(n_splits=5, shuffle=True)

for train_ix, test_ix in kfold.split(train_x):
    train_X, test_X=train_x.iloc[train_ix,:],train_x.iloc[test_ix,:]
    train_Y, test_Y=train_y[train_ix],train_y[test_ix]
    data_x=data_x.append(test_X)
    data_y.extend(test_Y)
    
    m1=NB_model
    m1.fit(train_X,train_Y)
    ypred=m1.predict(test_X)
    y_pred=y_pred.append(pd.DataFrame(ypred))
    
    m2=voting_clf

    m2.fit(train_X,train_Y)
    ypred_2=m2.predict(test_X)
    y_pred_2=y_pred_2.append(pd.DataFrame(ypred_2))

   
    
# the second stack (light gbm)

stack_x=data_x
stack_x["predit_rst"]=y_pred.values[:,0]
stack_x["rst2"]=y_pred_2.values[:,0]
stack_model = lightgbm_model_selected


rst1=NB_model.predict(x_valid)
rst2=voting_clf.predict(x_valid)
x_valid_stack=deepcopy(x_valid)

x_valid_stack["rst"]=rst1
x_valid_stack["rst2"]=rst2


stack_model.fit(stack_x,data_y)

Performance_Metrics(stack_model,"Stacking Classifer",x_valid_stack,y_valid)




