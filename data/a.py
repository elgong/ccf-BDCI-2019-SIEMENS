from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import math
import copy as cp
df1=pd.read_csv("first_round_training_data.csv")
df2=pd.read_csv("first_round_testing_data.csv")
train_x=np.zeros([6000,10],dtype=np.float32)
train_y=np.zeros([6000],dtype=np.int32)

test_id=np.zeros([6000],dtype=np.int32)
test_x=np.zeros([6000,10],dtype=np.float32)
cls2int={"Excellent":0,"Good":1,"Pass":2,"Fail":3}
for i in range(1,11):
    par="Parameter"+str(i)
    tmp=df1[par]
    for j in range(6000):
        train_x[j,i-1]=tmp[j]
        cls=cls2int[df1["Quality_label"][j]]
        train_y[j]=cls
for i in range(1,11):
    par="Parameter"+str(i)
    tmp=df2[par]
    for j in range(6000):
        test_x[j,i-1]=tmp[j]
        ID=int(df2["Group"][j])
        test_id[j]=ID
params = {
'boosting_type': 'gbdt',
'objective': 'multiclassova',
'num_class': 4,  
'metric': 'multi_error', 
'num_leaves': 63,
'learning_rate': 0.01,
'feature_fraction': 0.9,
'bagging_fraction': 0.9,
'bagging_seed':0,
'bagging_freq': 1,
'verbose': -1,
'reg_alpha':1,
'reg_lambda':2,
'lambda_l1': 0,
'lambda_l2': 1,
'num_threads': 8,
}
lgb_train = lgb.Dataset(train_x, train_y)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1300,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=100,
                )
f=open("lgb1300round.csv","w")
ans=gbm.predict(test_x, num_iteration=1300)
tmp=np.zeros([120,4])
cnt=np.zeros([120])
for i in range(6000):
    ID=test_id[i]
    tmp[ID,:]+=ans[i,:]
    cnt[ID]+=1
for i in range(120):
    SUM=np.sum(tmp[i,:])
    tmp[i,:]/=SUM

f.write("Group,Excellent ratio,Good ratio,Pass ratio,Fail ratio\n")
for i in range(120):
    f.write(str(i))
    for j in range(4):
        f.write(","+str(tmp[i,j]))
    f.write("\n")
f.close()