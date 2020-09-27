# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import data_cleaning as dc
import feature_engineering as fe


'''
Data Cleaning
'''
# use train/test csv file merged and cleaned by data cleaning.py

train_data=pd.read_csv("train_data_cleaned.csv")
test_data=pd.read_csv("test_data_cleaned.csv")

# import subset data

bureau=dc.data_input("bureau.csv")
    
bureau_balance=dc.data_input("bureau_balance.csv")
#merge bureau balance with bureau
bureau=bureau.merge(bureau_balance,on="SK_ID_BUREAU")

pre_app=dc.data_input("previous_application.csv")

pos_cash_balance=dc.data_input("POS_CASH_balance.csv")

credit_card_balance=dc.data_input("credit_card_balance.csv")

install_pay=dc.data_input("installments_payments.csv")

dic_add_on_dataset={}
dic_add_on_dataset.update({"bureau":bureau,"pre_application":pre_app,\
                   "pos_cash_balance":pos_cash_balance,"install_payment":install_pay,\
                    "credit_card_balance":credit_card_balance})


'''
Feature Engineering
'''

# add new features and combined with main table

feature_adder=fe.Main_feature_add()

# feature adder classes
l_feature_class=[fe.Bureau_AttributesAdder, fe.Pre_app_AttributesAdder,\
                fe.Pos_cash_AttributesAdder,fe.Install_pay_AttributesAdder,\
                fe.Credit_Card_AttributesAdder]


    
train_data=feature_adder.transform(train_data,l_feature_class,dic_add_on_dataset)

test_data=feature_adder.transform(test_data,l_feature_class,dic_add_on_dataset)

# create polynomial features from main table
feature_poly=fe.Polynomial_AttributesAdder()

#poly feature columns
data_poly_train=feature_poly.fit(train_data).feature
data_poly_test=feature_poly.fit(test_data).feature

# conbime all new features
train_data=pd.concat([train_data,data_poly_train],axis=1)
test_data=pd.concat([test_data,data_poly_test],axis=1)

# Drop highly correlated features
drop_feature_train=fe.Drop_correlated_features(train_data)
drop_feature_test=fe.Drop_correlated_features(test_data)

train_data=train_data.drop(columns=drop_feature_train)
test_data=test_data.drop(columns=drop_feature_test)

#save to csv train&test data added feature before transformation and one-hot encoding

train_data.to_csv("train_data_feature.csv",index=False)
test_data.to_csv("test_data_feature.csv",index=False)



