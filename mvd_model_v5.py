#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:56:38 2019

@author: mark
"""

import os
import glob
import pandas as pd
import numpy as np
import h2o
import datetime
h2o.init()

def d_frame():
    path = r'/media/mark/DE90-5EEA/'
    allFiles = glob.glob(path+ "/*.csv")


    np_array_list = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        np_array_list.append(df.as_matrix())

    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)

    big_frame.columns = ['Interaction ID',
                 'Date',
                 'Unit',
                 'Service Category',
                 'Service',
                 'Time_Queued',
                 'Time_Called',
                 'Wait_Time',
                 'Time_Attended',
                 'Time_Completed',
                 'Service_Time',
                 'Served_By_Userid',
                 'Ticket_Number',
                 'Service_Location']
    
    frame = big_frame[['Date', 'Unit', 'Service', 'Wait_Time']]

    return frame
    

def variables(frame):
    frame['Week'] = pd.DatetimeIndex(frame['Date']).week
    frame['Year'] = pd.DatetimeIndex(frame['Date']).year
    frame.drop(['Date'],axis=1,inplace=True)
    return frame

#    frame.drop(['Date'],axis=1,inplace=True)
#    return frame
    

def wait_convert(frame):
    minute = frame['Wait_Time'].apply(lambda x: x.split(":")[0]).astype(float)
    second = frame['Wait_Time'].apply(lambda x: x.split(":")[1]).astype(float)
    mil_sc = frame['Wait_Time'].apply(lambda x: x.split(":")[2]).astype(float)
    minute = minute * 60
    mil_sc = mil_sc /1000
    Wait_Time = minute + second + mil_sc
    frame.drop(['Wait_Time'], axis = 1, inplace = True)
    concatresult = pd.concat([frame, Wait_Time], axis=1)
    
    return concatresult


# Get one hot encoding of columns B
def dummies(frame):
    frame = pd.DataFrame(frame.pivot_table(index=['Unit'], columns = ['Service'], aggfunc=[len], fill_value=0))
    frame[frame>1] = 1
    frame.reset_index(level=0, inplace=True)
    frame.columns = frame.columns.droplevel(0) #remove len
    frame.columns = frame.columns.droplevel(0) #remove wait_Time
    frame.columns.name = None
    frame = frame.reset_index(drop=True)
    frame.rename(columns={'':'Unit'}, inplace=True)
    frame = frame.loc[:,~frame.columns.duplicated()]
    
    colnames = list(frame)
    for col in colnames:
        frame[col] = frame[col].astype('category')

    return frame
    


def get_frame(frame,dummies_df):
    frame = frame[frame.Wait_Time > .1]
    frame = frame.groupby(['Unit','Week','Year'])['Wait_Time'].mean().reset_index()
    z = pd.merge(frame,dummies_df, on='Unit', how='inner')
    z = z.groupby(["Year","Week"]).apply(lambda x: x.sort_values(["Unit"], ascending = False)).reset_index(drop=True)
    
    return z
    
def get_week(x):
    if x['Week'].iloc[-1] == 52:
        x.Week = 1 
        x.Year = x.Year + 1
    elif x['Week'].iloc[-1] < 52:
        x.Week = x['Week'].iloc[-1] + 1
    else:
        return 0
    
#def get_week(x):
#    if max(x.Week) == 52:
#        x.Week = 1
#        x.Year = x.Year + 1
#    elif max(x.Week) < 52:
#        x.Week = max(x.Week) + 1
#    else:
#        return 0

def test_set(frame):
    z = frame[frame['Week']==frame['Week'].iloc[-1]]
    
    get_week(z)
    z.reset_index(drop=True)
    
    y = z.copy()
    get_week(y)

    v = y.copy()
    get_week(v)

    w = v.copy()
    get_week(w)
    
    frames = [z,y,v,w]
    con = pd.concat(frames)
    con.Wait_Time = 0
    
    return con
    
    

    

def data_fix(train,test):
    train.Week = train.Week.astype('category')
    train.Year = train.Year.astype('category')
    test.Week  = test.Week.astype('category')
    test.Year  = test.Year.astype('category')
    return train,test
    

def model(train,test):

    today = datetime.datetime.today().today().strftime('%Y-%m-%d:%H:%M')
    
    from h2o.estimators import H2OGeneralizedLinearEstimator
    
    h2o_train = h2o.H2OFrame(train)
    h2o_test  = h2o.H2OFrame(test)
    

    predictor_columns = [c for c in h2o_train.drop('Wait_Time').col_names if c not in 'Unit']
    response_column = 'Wait_Time'

    h2o_train[predictor_columns] = h2o_train[predictor_columns].asfactor()
    h2o_test[predictor_columns]  = h2o_test[predictor_columns].asfactor()

 #   train, valid = h2o_train.split_frame([.99],seed=615)


    glm_model = H2OGeneralizedLinearEstimator(family = 'Gamma', #Gaussian , Gamma
                                          lambda_= 0,
                                          alpha = 0,
                                          compute_p_values = True,
                                          remove_collinear_columns=True,
                                          seed = 615,
                                          fold_assignment = "Modulo",   ### "Modulo"
                                          keep_cross_validation_predictions = True,
                                          nfolds = 7)

    glm_model.train(predictor_columns, response_column, training_frame=h2o_train, validation_frame = h2o_test)

    glm_model.model_performance(h2o_train)
    glm_model.model_performance(h2o_test)

    prediction = glm_model.predict(h2o_test).as_data_frame()
    prediction['pred_min'] = (prediction.predict/60) * 10
    prediction['StdErr_min'] = (prediction.StdErr/60) 
    pred_table = test[['Unit', 'Week']].merge(prediction, how='outer', left_index = True, right_index=True)

    coef_table = glm_model._model_json['output']['coefficients_table'].as_data_frame()
    
    pred_table[pred_table.Unit == 'Essex']

    coef_table.to_csv('/home/mark/Desktop/IB_docs/coef_table' + today + '.csv',index=False)
    pred_table.to_csv('/home/mark/Desktop/IB_docs/pred_table' + today + '.csv',index=False)
    return
    
    

    
def main():
    frame = d_frame()
    frame = variables(frame)
    frame = wait_convert(frame)
    dummies_df = dummies(frame)
    train = get_frame(frame, dummies_df)
    test = test_set(train).reset_index(drop=True)
    train,test = data_fix(train,test)
    model(train,test)
    
    h2o.cluster().shutdown()

    

if __name__ == '__main__':
    main()
