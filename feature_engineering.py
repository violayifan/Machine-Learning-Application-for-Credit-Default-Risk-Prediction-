# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:46:56 2020

@author: Viola
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#Part 1 create new features 

#1.create features use domain knowledge

# 1.1 from Bureau

class Bureau_AttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self,add_bureau_attr=True):
        self.add_bureau_attr=add_bureau_attr
    
    def fit(self,bureau):
        
        bureau_add=bureau.copy()
        bureau_add['bureau_debt_credit_ratio'] = \
        bureau_add['AMT_CREDIT_SUM_DEBT'] / bureau_add["AMT_CREDIT_SUM"]
        
        bureau_add['bureau_overdue_debt_ratio'] = \
        bureau_add['AMT_CREDIT_SUM_OVERDUE'] / bureau_add['AMT_CREDIT_SUM_DEBT']
              
        bureau_add_feature=bureau_add.loc[:,["SK_ID_CURR","bureau_debt_credit_ratio","bureau_overdue_debt_ratio"]]
        
        self.feature=bureau_add_feature
        self.bureau_add=bureau_add
        return self
    def transform(self,bureau):
        if self.add_bureau_attr :
            self.fit(bureau)
            Bureau_add=self.bureau_add
            return Bureau_add
            

# 1.2 from Credit Card Balance
            
class Credit_Card_AttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self,add_credit_attr=True):
        self.add_credit_attr=add_credit_attr
    
    def fit(self,credit_card_balance):
        
        credit_add=credit_card_balance.copy()
        #Drawings/limit
        credit_add["draw_limit_ratio"]=\
        credit_add["AMT_DRAWINGS_CURRENT"]/credit_add["AMT_CREDIT_LIMIT_ACTUAL"]
        
        #Special expense
        
        credit_add["special_expense"]=credit_add["CNT_DRAWINGS_CURRENT"]- \
        (credit_add["CNT_DRAWINGS_ATM_CURRENT"]+credit_add["CNT_DRAWINGS_OTHER_CURRENT"]+\
         credit_add["CNT_DRAWINGS_POS_CURRENT"])
        
        credit_add_feature=credit_add.loc[:,["SK_ID_CURR","draw_limit_ratio","special_expense"]]
        
        self.feature=credit_add_feature
        self.credit_add=credit_add
        return self
    def transform(self,credit_card_balance):
        if self.add_credit_attr :
            self.fit(credit_card_balance)
            credit_add=self.credit_add
            return credit_add

# 1.3 from Previous application

class Pre_app_AttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self,add_pre_app_attr=True):
        self.add_pre_app_attr=add_pre_app_attr
    
    def fit(self,pre_application):
        
        pre_app_add=pre_application.copy()
        
        # credit application ratio
        pre_app_add["credit_application_ratio"]=pre_app_add["AMT_CREDIT"]/pre_app_add["AMT_APPLICATION"]
              
        pre_app_add_feature=pre_app_add.loc[:,["SK_ID_CURR","credit_application_ratio"]]
        
        self.feature=pre_app_add_feature
        self.pre_app_add=pre_app_add
        return self
    def transform(self,pre_application):
        if self.add_pre_app_attr :
            self.fit(pre_application)
            pre_app_add=self.pre_app_add
            return pre_app_add
    
# 1.4 from Pos cash balance
            
class Pos_cash_AttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self,add_pos_cash_attr=True):
        self.add_pos_cash_attr=add_pos_cash_attr
    
    def fit(self,pos_cash_balance):
        
        pos_cash_add=pos_cash_balance.copy()
        #installment ratio
        pos_cash_add["installment_ratio"]=pos_cash_add["CNT_INSTALMENT_FUTURE"]/pos_cash_add["CNT_INSTALMENT"]
              
        pos_cash_add_feature=pos_cash_add.loc[:,["SK_ID_CURR","installment_ratio"]]
        
        self.feature= pos_cash_add_feature
        self.pos_cash_add=pos_cash_add
        return self
    def transform(self,pos_cash_balance):
        if self.add_pos_cash_attr :
            self.fit(pos_cash_balance)
            pos_cash_add=self.pos_cash_add
            return  pos_cash_add
    
# 1.5 from Install payment

class Install_pay_AttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self,add_install_pay_attr=True):
        self.add_install_pay_attr=add_install_pay_attr
    
    def fit(self,install_pay):
        
        install_pay_add=install_pay.copy()
        #days overdue
        install_pay_add["days_overdue"]= install_pay_add["DAYS_ENTRY_PAYMENT"]- install_pay_add["DAYS_INSTALMENT"]
        
        #payment install ratio
        install_pay_add["payment_install_ratio"]=install_pay_add["AMT_INSTALMENT"]/install_pay_add["AMT_PAYMENT"]
        
        install_pay_add_feature=install_pay_add.loc[:,["SK_ID_CURR","days_overdue","payment_install_ratio"]]
        
        self.feature=install_pay_add_feature
        self.install_pay_add=install_pay_add
        return self
    def transform(self,install_pay):
        if self.add_install_pay_attr :
            self.fit(install_pay)
            install_pay_add=self.install_pay_add
            return install_pay_add
        


#2.create polynomial features from main table

class Polynomial_AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_Polynomial_attr=True):
        self.add_Polynomial_attr=add_Polynomial_attr
    
    def fit(self,t_data):
        
        poly_features = t_data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
        #poly_target= t_data["TARGET"]
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        #impute nan value
        poly_features=imp_mean.fit_transform(poly_features)
        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree = 3)
        
        poly_transformer.fit(poly_features)

        # Transform the features
        poly_features = poly_transformer.transform(poly_features)
        
        poly_feature_name=poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', \
                                       'EXT_SOURCE_2', 'EXT_SOURCE_3','DAYS_BIRTH'])
        df_poly_feature=pd.DataFrame(poly_features,columns=poly_feature_name).iloc[:,5:]
        
        t_data_add=pd.concat([t_data,df_poly_feature],axis=1)
        
        self.feature= df_poly_feature
        self.polynomial_add_data= t_data_add
        return self
    def transform(self,t_data):
        if self.add_Polynomial_attr :
            self.fit(t_data)
            t_data_add=self.polynomial_add_data
            return t_data_add
        

#Part 2 merge new features with the main cleaned dataset
class Main_feature_add(BaseEstimator, TransformerMixin):
    def __init__(self,add_all_attr=True):
        
        self.add_all_attr=add_all_attr
    
    def fit(self,t_data,f_class,f_data):
        # *arg is the add feature class
        # **arg is the original sub dataset of the feature
       t_data_add=t_data.copy()
       
       dic_feature={}
       for feature_class,feature_data in zip (f_class,f_data.items()):
           add_feature= feature_class()
           add_feature.fit(feature_data[1])
           feature=add_feature.feature
           dic_feature[feature_data[0]]=feature
           print("{} completed".format(feature_data[0]))
        
       feature=pd.concat(dic_feature.values()).groupby("SK_ID_CURR").mean()
       
       t_data_add=t_data_add.merge(feature,on="SK_ID_CURR",sort=False,how="left")
           
       self.data_new_feature=t_data_add
       
       return self
    
    def transform(self,t_data,f_class,f_data):
        if self.add_all_attr :
            self.fit(t_data,f_class,f_data)
            t_data_add=self.data_new_feature
            return t_data_add


#Drop highly correlated features
            
def Drop_correlated_features(data,threshold=0.8):
    
    #return drop feature column  
    corr_matrix=data.corr().abs()
    upper_tri_matrix=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_feature=[feature_col for feature_col in upper_tri_matrix.columns if any(upper_tri_matrix[feature_col] > threshold)]
    return drop_feature
    
    
#Reduce dimension using PCA
    
def PCA_feature(data):
    pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')),
             ('pca', PCA(n_components=0.9))])
    
    data_pca=pipeline.fit_transform(data)
    
    return data_pca

def PCA_plot(data):
    
    pca = PCA()
    pca.fit(data)
    
    # Plot the cumulative variance explained

    plt.figure(figsize = (10, 8))
    plt.plot(list(range(data.shape[1])), np.cumsum(pca.explained_variance_ratio_), 'b-')
    plt.xlabel('Number of PC'); plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained with PCA')

    
    

    






