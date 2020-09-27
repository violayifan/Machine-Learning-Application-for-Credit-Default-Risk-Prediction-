# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:37:16 2020

@author: Viola
"""


import numpy as np
import pandas as pd
from datetime import datetime
import logging
from collections import Counter
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger('data_cleaning')  
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('data_cleaning.log')  
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
fh.setFormatter(formatter)  
ch.setFormatter(formatter)  
logger.addHandler(fh)  
logger.addHandler(ch)  


def data_input(file):
    return pd.read_csv(file,header=0)


def data_merge(add_on):
    #process merged data
    #deal with muti id record
    
    #seperate numerical and categorical features
    add_on_temp=add_on.iloc[:,2:]
    add_on_1=add_on_temp._get_numeric_data()
    prob_num_col=set(add_on_1.columns)
    
    #exclude categorical label
    ex_col=[]
    for col in prob_num_col:
        value=add_on_1[col].copy().dropna().unique()
        if len(value)==2 and (0 in value and 1 in value):
            ex_col.append(col)
            
        if len(value[value<0])==0:
            value=pd.Series(np.sort(value))
            diff=value.diff()
            if 1 in diff.values:
                if diff.value_counts()[1]/len(diff)>=0.9:
                    ex_col.append(col)
    
    num_col=list(prob_num_col-set(ex_col))
    num_col.append("SK_ID_CURR")
    
    add_on_1=add_on.loc[:,num_col]
    
    add_on_1=add_on_1.groupby("SK_ID_CURR").mean().reset_index()
    
    #categorical label
    cat_col=list(set(add_on_temp.columns)-set(num_col))
    cat_col.append("SK_ID_CURR")
    
    add_on_2=add_on.loc[:,cat_col].set_index("SK_ID_CURR")
    add_on_2.dropna(inplace=True)
    add_on_2=add_on_2.groupby("SK_ID_CURR").transform(lambda x: Counter(x).most_common(1)[0][0])
        
    add_on_2=add_on_2.reset_index().drop_duplicates(subset="SK_ID_CURR")
              
    add_on_new=add_on_1.merge(add_on_2,on="SK_ID_CURR",sort=False)
    
    return add_on_new



class data_cleaner(object):
    
    def main(self,dic_dataset):
        
        dic_dataset.update(self.bureau_data_clean(dic_dataset["bureau"]))
        dic_dataset.update(self. pre_app_clean(dic_dataset["pre_application"]))
        dic_dataset.update(self.credit_card_balance_clean(dic_dataset["credit_card_balance"]))
        return dic_dataset
        
    
    def drop_na_feature(self,train_data,test_data,threshold=0.7):
        
        #detect percentage of nan value
        train_missing=train_data.isnull().sum()/len(train_data)
        test_missing=test_data.isnull().sum()/len(test_data)
        
        train_drop=set(train_missing.index[train_missing > threshold])
        test_drop=set(test_missing.index[test_missing > threshold])        
        self.drop_feature=list(set(train_drop|test_drop))
        
        
    
    def main_data_clean(self,main_data):
        
        #drop nan value if percentage > 0.70 
        main_data=main_data.drop(columns=self.drop_feature)
        
        #deal with anomalies
        main_data['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        main_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        main_data['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        main_data['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        main_data['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
        return main_data
        
        
        
    def bureau_data_clean(self,bureau):
        
        #deal with anomalies    
        bureau[bureau.DAYS_CREDIT_ENDDATE<-40000]= np.nan
        bureau[bureau.DAYS_CREDIT_UPDATE < -40000] = np.nan
        bureau[bureau.DAYS_ENDDATE_FACT  < -40000] = np.nan
        
        return {"bureau":bureau}
        
    def pre_app_clean(self,pre_app):
        
        pre_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        pre_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        pre_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan,inplace=True)
        pre_app['DAYS_LAST_DUE'].replace(365243, np.nan,inplace=True)
        pre_app['DAYS_TERMINATION'].replace(365243, np.nan,inplace=True)
        return {"pre_application":pre_app}
    
    def credit_card_balance_clean(self,credit_card_balance):
    
        credit_card_balance[credit_card_balance.AMT_DRAWINGS_ATM_CURRENT< 0] = np.nan
        credit_card_balance[credit_card_balance.AMT_DRAWINGS_CURRENT< 0] = np.nan
 
        return {"credit_card_balance":credit_card_balance}
    
        
    
class data_transformer(object):
    
    
    def trans_cat(self,data):
    #label encoding & fillna
        label_encoder = preprocessing.LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object':
            # If 2 or fewer unique categoricalries
                if len(list(data[col].unique())) == 2 and isinstance(data[col].unique()[0],str) and\
                    isinstance(data[col].unique()[1],str):
                    data[col]=label_encoder.fit_transform(data[col])
                    data[col].fillna(-1,inplace=True)
        return data
        
    def trans_num(self,data):      
    #standardized and fillna
        num_pipeline = Pipeline([('imputer', SimpleImputer(missing_values=np.nan,strategy="median")),\
                             ('std_scaler', preprocessing.StandardScaler())])
        data.replace([np.inf, -np.inf], np.nan,inplace=True)
        for col in data.columns:
            if data[col].dtype != "object" :
                data[[col]]=num_pipeline.fit_transform(data[[col]])
        return data
    
    

if __name__=="__main__":
    
    '''
    Preprocessing result excluding data transformation
    '''
    
    #data input
    
    train_data=data_input("application_train.csv")
    test_data=data_input("application_test.csv")
    
    bureau=data_input("bureau.csv")
    
    bureau_balance=data_input("bureau_balance.csv")
    #merge bureau balance with bureau
    bureau=bureau.merge(bureau_balance,on="SK_ID_BUREAU")
    
    pre_app=data_input("previous_application.csv")
    
    pos_cash_balance=data_input("POS_CASH_balance.csv")
    
    credit_card_balance=data_input("credit_card_balance.csv")
    
    install_pay=data_input("installments_payments.csv")
    
    logger.info("data input completed")

    dic_add_on_dataset={}
    dic_add_on_dataset.update({"bureau":bureau,"pre_application":pre_app,\
                       "pos_cash_balance":pos_cash_balance,"install_payment":install_pay,\
                        "credit_card_balance":credit_card_balance})
    
    
    # clean add_on data   
    cleaner= data_cleaner()
    dic_add_on_dataset=cleaner.main(dic_add_on_dataset)
    
        
    #deal with multi id
    for key,add_on in dic_add_on_dataset.items():
        start=datetime.now()
        dic_add_on_dataset[key]=data_merge(add_on)
        logger.info('merge {}'.format(key))
        end=datetime.now()
        logger.info("processing merged data time cost: {}".format(end-start))
        
    #merge with main data
    for key,add_on in dic_add_on_dataset.items():
        train_data=train_data.merge(add_on,on="SK_ID_CURR",sort=False,how="left")
        test_data=test_data.merge(add_on,on="SK_ID_CURR",sort=False,how="left")
        logger.info("Merged data : {}".format(key))

        
    # drop nan feature & main data clean
    cleaner.drop_na_feature(train_data,test_data)
    
    train_data=cleaner.main_data_clean(train_data)
    test_data=cleaner.main_data_clean(test_data)
    
  
    train_data.to_csv("train_data_cleaned.csv",index=False)
    test_data.to_csv("test_data_cleaned.csv",index=False)
    
    
    
        
    
    
    
    
    
    
    