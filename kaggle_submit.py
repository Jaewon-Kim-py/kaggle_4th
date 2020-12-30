#!/usr/bin/env python
# coding: utf-8

# 데이터 설명
# 
# train/test는 14개의 columns으로 구성되어 있고, train은 예측해야 하는 target 값 feature까지 1개가 추가로 있습니다. 각 데이터는 다음을 의미합니다.
# 
# id
# age : 나이
# workclass : 고용 형태
# fnlwgt : 사람 대표성을 나타내는 가중치 (final weight의 약자)
# education : 교육 수준
# education_num : 교육 수준 수치
# marital_status: 결혼 상태
# occupation : 업종
# relationship : 가족 관계
# race : 인종
# sex : 성별
# capital_gain : 양도 소득
# capital_loss : 양도 손실
# hours_per_week : 주당 근무 시간
# native_country : 국적
# income : 수익 (예측해야 하는 값)
# >50K : 1
# <=50K : 0

# 데이터 불러오기

# In[1]:


import warnings

warnings.filterwarnings(action='ignore')

from os.path import join
import pandas as pd
import numpy  as np
import xgboost as xgb


sample_data = pd.read_csv(join('data', join('kaggle', 'train.csv')))


# In[2]:


sample_data


# 파생변수 생성
# capital  = capital_gain - capital_loss
# 
# age_group 설정
# 
# country_group 설정
# 
# education_group 설정

# In[ ]:


#파생변수 생성(capital)
sample_data['capital'] = sample_data['capital_gain'] - sample_data['capital_loss']


# In[ ]:


sample_data.drop(columns=['capital_gain','capital_loss'],inplace=True)


# In[ ]:


#파생변수 생성(age_group)
bins = [0, 29, 39, 64, 120]
labels = ['youth', 'middle', 'middle_elder', 'elder']
sample_data['age_group'] = pd.cut(sample_data.age, bins, labels = labels,include_lowest = True)


# In[ ]:


sample_data.drop(columns=['age'],inplace=True)


# In[ ]:


#파생변수 생성(country_group)
condition = [
    sample_data['native_country'].str.contains('United-States|Canada', na=False),
    sample_data['native_country'].str.contains('Mexico|Cuba|El-Salvador|Guatemala|Columbia|Peru|Nicaragua|Honduras|Ecuador', na=False),
    sample_data['native_country'].str.contains('Poland|Germany|France|Ireland|England|Italy|Scotland|Holand-Netherlands', na=False),
    sample_data['native_country'].str.contains('Portugal|Hungary|Greece|Yugoslavia', na=False),
    sample_data['native_country'].str.contains('Philippines|Vietnam|Thailand|Laos|Cambodia', na=False),
    sample_data['native_country'].str.contains('Taiwan|Japan|China|Hong|South', na=False),
    sample_data['native_country'].str.contains('Iran|India', na=False),
    sample_data['native_country'].str.contains('Puerto-Rico|Haiti|Trinadad&Tobago|Outlying-US(Guam-USVI-etc)', na=False),
    sample_data['native_country'].str.contains('Jamaica|Dominican-Republic', na=False)
]

choices = ['N_America', 'S_America', 'High_Europe', 'Mid_Europe', 'ES_Asia', 'E_Asia', 'W_Asia', 'Used_Colony', 'Africa']

sample_data['country_group'] = np.select(condition, choices, default = 'United-States')


# In[ ]:


sample_data.drop(columns=['native_country'], inplace=True)


# In[ ]:


#education 변수
sample_data['education'].unique()


# In[ ]:


condition = [
    sample_data['education'].str.contains('Preschool',na=False),
    sample_data['education'].str.contains('1st-4th|5th-6th',na=False),
    sample_data['education'].str.contains('7th-8th|9th',na=False),
    sample_data['education'].str.contains('10th|11th|12th',na=False),
    sample_data['education'].str.contains('HS-grad',na=False),
    sample_data['education'].str.contains('Some-college',na=False),
    sample_data['education'].str.contains('Assoc-acdm|Assoc-voc',na=False),
    sample_data['education'].str.contains('Bachelors',na=False),
    sample_data['education'].str.contains('Masters',na=False),
    sample_data['education'].str.contains('Prof-school|Doctorate',na=False)
]
choices = ['pre','elementry','middleschool','highschool','highschool_grad','college','assoc','bachelors','ms','ph.d']
sample_data['education_level'] = np.select(condition, choices, default = 'others')


# In[ ]:


sample_data.drop(columns=['education','education_num'],inplace=True)


# 데이터 확인

# In[ ]:


sample_data


# #미리 제거하거나 분리해야할 변수 설정
# (income의 경우 target, id 경우 필요없는 변수, workclass,occupation의 경우 따로 예측할
# 변수이므로)

# In[ ]:


label = sample_data['income']
workclass = sample_data['workclass']
occupation = sample_data['occupation']


# In[ ]:


sample_data.drop(columns=['id','workclass','occupation','income'], inplace=True)


# 연속형, 범주형 변수 구분

# In[ ]:


cat_columns = [c for (c, t) in zip(sample_data.dtypes.index, sample_data.dtypes) if t == 'O'] 
num_columns = [c for c in sample_data.columns if c not in cat_columns]


# In[ ]:


sample_data['age_group'] = sample_data['age_group'].astype('object')


# In[ ]:


cat_columns = [c for (c, t) in zip(sample_data.dtypes.index, sample_data.dtypes) if t == 'O'] 
num_columns = [c for c in sample_data.columns if c not in cat_columns]


# In[ ]:


sample_data_num = sample_data[num_columns]


# In[ ]:


sample_data_cat = sample_data[cat_columns]


# 범주형 변수의 one-hot encoding 실시

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
sample_data_cat = ohe.fit_transform(sample_data_cat)


# In[ ]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[ ]:


sample_data_cat = pd.DataFrame(sample_data_cat,columns = ohe_columns)


# 연속형, 범주형 분리한 것 합치기
# income, occupation 붙이기(결측치에 대해 예측하여야 하므로 따로 분리 후 합침)

# In[ ]:


sample_data_num.reset_index(drop=True)
sample_data_cat.reset_index(drop=True)
sample_data1 = pd.concat([sample_data_num,sample_data_cat],axis=1)


# In[ ]:


sample_data1.reset_index(drop=True)
label = label.reset_index(drop=True)
sample_data1 = pd.concat([sample_data1,label],axis=1)


# In[ ]:


sample_data1.reset_index(drop=True)
occupation = occupation.reset_index(drop=True)
sample_data1 = pd.concat([sample_data1,occupation],axis=1)


# In[ ]:


sample_data1


# workclass 붙이기(occupation과 같은 이유)

# In[ ]:


workclass.reset_index(drop=True)
sample_data1.reset_index(drop=True)
sample_data_workclass = pd.concat([sample_data1,workclass],axis=1)


# In[ ]:


sample_data_workclass


# 예측 및 타겟변수 분리 실시

# In[ ]:


sample_data_null = sample_data_workclass[sample_data_workclass['workclass']=='?']
sample_data_notnull = sample_data_workclass[sample_data_workclass['workclass']!='?']


# In[ ]:


null_income = sample_data_null['income']
notnull_income = sample_data_notnull['income']
sample_data_null.drop(columns=['income'],inplace=True)
sample_data_notnull.drop(columns=['income'],inplace=True)


# In[ ]:


null_occupation = sample_data_null['occupation']
notnull_occupation = sample_data_notnull['occupation']
sample_data_null.drop(columns=['occupation'],inplace=True)
sample_data_notnull.drop(columns=['occupation'],inplace=True)


# In[ ]:


target1 = sample_data_notnull['workclass']
sample_data_null.drop(columns=['workclass'],inplace=True)
sample_data_notnull.drop(columns=['workclass'],inplace=True)


# workclass='?' 에 대한 예측 모델 생성

# In[ ]:


sample_data_null1 = sample_data_null.copy()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(sample_data_notnull, target1, 
                                                      test_size=0.3,
                                                      random_state=2020,
                                                      shuffle=True)


# In[ ]:


x_train_mean = np.mean(x_train[num_columns], axis=0)
x_train_std  = np.std(x_train[num_columns], axis=0)


x_train.loc[:, num_columns] = (x_train[num_columns] - x_train_mean) / x_train_std 
x_valid.loc[:, num_columns] = (x_valid[num_columns] - x_train_mean) / x_train_std
sample_data_null1.loc[:, num_columns] = (sample_data_null1[num_columns] - x_train_mean) / x_train_std


# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[ ]:


xgb.fit(x_train,y_train)


# In[ ]:


workclass_pred = xgb.predict(sample_data_null1)


# In[ ]:


col = ['workclass']


# In[ ]:


workclass_pred = pd.DataFrame(workclass_pred,columns=col)


# 합치기(1)- sample_data_null 과 workclass_pred

# In[ ]:


workclass_pred = workclass_pred.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, workclass_pred], axis=1)


# In[ ]:


sample_data_null = sample_data_null.reset_index(drop=True)
null_income = null_income.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, null_income], axis=1)


# In[ ]:


sample_data_null = sample_data_null.reset_index(drop=True)
null_occupation = null_occupation.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,null_occupation],axis=1)


# In[ ]:


sample_data_null


# 합치기(2) sample_data_notnull

# In[ ]:


sample_data_notnull.reset_index(drop=True)
target1.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,target1],axis=1)


# In[ ]:


sample_data_notnull


# income 합치기

# In[ ]:


sample_data_notnull.reset_index(drop=True)
notnull_income.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,notnull_income],axis=1)


# In[ ]:


sample_data_notnull


# occupation 합치기

# In[ ]:


sample_data_notnull.reset_index(drop=True)
notnull_occupation.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,notnull_occupation],axis=1)


# In[ ]:


sample_data_notnull


# 합치기

# In[ ]:


sample_data_notnull.reset_index(drop=True)
sample_data_null.reset_index(drop=True)
sample_data = pd.concat([sample_data_notnull,sample_data_null],axis=0)


# In[ ]:


sample_data


# test.csv에 대한 전처리

# In[ ]:


x_test = pd.read_csv(join('data', join('kaggle', 'test.csv')))


# In[ ]:


x_test['capital'] = x_test['capital_gain'] - x_test['capital_loss']


# In[ ]:


x_test.drop(columns=['capital_gain','capital_loss'],inplace=True)


# In[ ]:


#파생변수 생성(age_group)
bins = [0, 29, 39, 64, 120]
labels = ['youth', 'middle', 'middle_elder', 'elder']
x_test['age_group'] = pd.cut(x_test.age, bins, labels = labels,include_lowest = True)


# In[ ]:


x_test.drop(columns=['age'],inplace=True)


# In[ ]:


#파생변수 생성(country_group)
condition = [
    x_test['native_country'].str.contains('United-States|Canada', na=False),
    x_test['native_country'].str.contains('Mexico|Cuba|El-Salvador|Guatemala|Columbia|Peru|Nicaragua|Honduras|Ecuador', na=False),
    x_test['native_country'].str.contains('Poland|Germany|France|Ireland|England|Italy|Scotland|Holand-Netherlands', na=False),
    x_test['native_country'].str.contains('Portugal|Hungary|Greece|Yugoslavia', na=False),
    x_test['native_country'].str.contains('Philippines|Vietnam|Thailand|Laos|Cambodia', na=False),
    x_test['native_country'].str.contains('Taiwan|Japan|China|Hong|South', na=False),
    x_test['native_country'].str.contains('Iran|India', na=False),
    x_test['native_country'].str.contains('Puerto-Rico|Haiti|Trinadad&Tobago|Outlying-US(Guam-USVI-etc)', na=False),
    x_test['native_country'].str.contains('Jamaica|Dominican-Republic', na=False)
]

choices = ['N_America', 'S_America', 'High_Europe', 'Mid_Europe', 'ES_Asia', 'E_Asia', 'W_Asia', 'Used_Colony', 'Africa']

x_test['country_group'] = np.select(condition, choices, default = 'United-States')


# In[ ]:


x_test.drop(columns=['native_country'], inplace=True)


# In[ ]:


condition = [
    x_test['education'].str.contains('Preschool',na=False),
    x_test['education'].str.contains('1st-4th|5th-6th',na=False),
    x_test['education'].str.contains('7th-8th|9th',na=False),
    x_test['education'].str.contains('10th|11th|12th',na=False),
    x_test['education'].str.contains('HS-grad',na=False),
    x_test['education'].str.contains('Some-college',na=False),
    x_test['education'].str.contains('Assoc-acdm|Assoc-voc',na=False),
    x_test['education'].str.contains('Bachelors',na=False),
    x_test['education'].str.contains('Masters',na=False),
    x_test['education'].str.contains('Prof-school|Doctorate',na=False)
]
choices = ['pre','elementry','middleschool','highschool','highschool_grad','college','assoc','bachelors','ms','ph.d']
x_test['education_level'] = np.select(condition, choices,  default = 'others')


# In[ ]:


x_test.drop(columns=['education','education_num'], inplace=True)


# In[ ]:


x_test


# In[ ]:


#변수 분리


# In[ ]:


workclass=x_test['workclass']
occupation = x_test['occupation']
id = x_test['id']
x_test.drop(columns=['workclass','occupation','id'],inplace=True)


# In[ ]:


#연속형 범주형 구분


# In[ ]:


cat_columns = [c for (c, t) in zip(x_test.dtypes.index, x_test.dtypes) if t == 'O'] 
num_columns = [c for c in x_test.columns if c not in cat_columns]


# In[ ]:


x_test['age_group'] = x_test['age_group'].astype('object')


# In[ ]:


cat_columns = [c for (c, t) in zip(x_test.dtypes.index, x_test.dtypes) if t == 'O'] 
num_columns = [c for c in x_test.columns if c not in cat_columns]


# In[ ]:


x_test_num = x_test[num_columns]
x_test_cat = x_test[cat_columns]


# In[ ]:


#범주형 인코딩


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
x_test_cat = ohe.fit_transform(x_test_cat)


# In[ ]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[ ]:


x_test_cat = pd.DataFrame(x_test_cat,columns = ohe_columns)


# In[ ]:


#num col 합치기


# In[ ]:


x_test_num.reset_index(drop=True)
x_test_cat.reset_index(drop=True)
x_test = pd.concat([x_test_num,x_test_cat],axis=1)


# In[ ]:


x_test


# In[ ]:


#workclass ocupation 붙이기


# In[ ]:


x_test = x_test.reset_index(drop=True)
workclass= workclass.reset_index(drop=True)
x_test = pd.concat([x_test,workclass],axis=1)


# In[ ]:


x_test.reset_index(drop=True)
occupation.reset_index(drop=True)
x_test = pd.concat([x_test,occupation],axis=1)


# In[ ]:


x_test.reset_index(drop=True)
id.reset_index(drop=True)
x_test = pd.concat([x_test,id],axis=1)


# In[ ]:


# original data
x_test


# In[ ]:


#occupation 분리


# In[ ]:


x_test_null = x_test[x_test['occupation']=='?']
x_test_notnull = x_test[x_test['occupation'] !='?']


# In[ ]:


x_test_null


# In[ ]:


x_test_notnull


# In[ ]:


workclass_null = x_test_null['workclass']
occupation_null = x_test_null['occupation']
id_null =x_test_null['id']
x_test_null.drop(columns=['workclass','occupation','id'],inplace=True)


# In[ ]:


#test 내용이 train에 섞이지 않도록 주의
predict = xgb.predict(x_test_null)


# In[ ]:


col = ['workclass']


# In[ ]:


predict = pd.DataFrame(predict,columns=col)


# In[ ]:


x_test_null = x_test_null.reset_index(drop=True)
predict = predict.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,predict],axis=1)


# In[ ]:


x_test_null


# In[ ]:


#occupation 붙이기


# In[ ]:


occupation_null = occupation_null.reset_index(drop=True)
x_test_null = x_test_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null, occupation_null],axis=1)


# In[ ]:


x_test_null


# In[ ]:


#id 붙이기


# In[ ]:


id_null = id_null.reset_index(drop=True)
x_test_null = x_test_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,id_null],axis=1)


# In[ ]:


#null notnull 합치기


# In[ ]:


x_test_null = x_test_null.reset_index(drop=True)
x_test_notnull = x_test_notnull.reset_index(drop=True)
x_test = pd.concat([x_test_null,x_test_notnull],axis=0)


# In[ ]:


x_test


# occupation 모델 만들기

# In[ ]:


sample_data


# #occupation ?  구분

# In[ ]:


sample_data_null = sample_data[sample_data['occupation']=='?']
sample_data_notnull = sample_data[sample_data['occupation'] !='?']


# In[ ]:


sample_data_null


# #변수 분리 및 제거

# In[ ]:


workclass_null = sample_data_null['workclass']
income_null = sample_data_null['income']
sample_data_null.drop(columns=['workclass','income','occupation'],inplace=True)


# In[ ]:


workclass_notnull = sample_data_notnull['workclass']
income_notnull = sample_data_notnull['income']
target = sample_data_notnull['occupation']
sample_data_notnull.drop(columns=['workclass','income','occupation'],inplace=True)


# #occupation 예측 모델 생성

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(sample_data_notnull, target, 
                                                      test_size=0.3,
                                                      random_state=2020,
                                                      shuffle=True)


# In[ ]:


sample_data_null1 = sample_data_null.copy()


# In[ ]:


x_train_mean = np.mean(x_train[num_columns], axis=0)
x_train_std  = np.std(x_train[num_columns], axis=0)


x_train.loc[:, num_columns] = (x_train[num_columns] - x_train_mean) / x_train_std 
x_valid.loc[:, num_columns] = (x_valid[num_columns] - x_train_mean) / x_train_std
sample_data_null1.loc[:, num_columns] = (sample_data_null1[num_columns] - x_train_mean) / x_train_std


# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[ ]:


xgb.fit(x_train,y_train)


# In[ ]:


occupation_pred = xgb.predict(sample_data_null1)


# In[ ]:


col = ['occupation']


# In[ ]:


occupation_pred = pd.DataFrame(occupation_pred,columns=col)


# #workclass 합치기

# In[ ]:


sample_data_null = sample_data_null.reset_index(drop=True)
workclass_null = workclass_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,workclass_null],axis=1)


# #합치기(1)- sample_data_null 과 occupation_pred

# In[ ]:


occupation_pred = occupation_pred.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,occupation_pred],axis=1)


# In[ ]:


sample_data_null


# #income 합치기

# In[ ]:


sample_data_null = sample_data_null.reset_index(drop=True)
income_null = income_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, income_null], axis=1)


# In[ ]:


sample_data_null


# sample_data_notnull 붙이기

# In[ ]:


sample_data_notnull.reset_index(drop=True)
workclass_notnull.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,workclass_notnull],axis=1)


# In[ ]:


sample_data_notnull.reset_index(drop=True)
target.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,target],axis=1)


# In[ ]:


sample_data_notnull.reset_index(drop=True)
income_notnull.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,income_notnull],axis=1)


# #sample_data_notnull sample_data_null 합치기

# In[ ]:


sample_data_notnull = sample_data_notnull.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data = pd.concat([sample_data_null,sample_data_notnull],axis=0)


# In[ ]:


sample_data


# #workclass occupation 인코딩 처리

# In[ ]:


encoding = sample_data[['workclass','occupation']]
sample_data.drop(columns=['workclass','occupation'],inplace=True)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
encoding = ohe.fit_transform(encoding)


# In[ ]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[ ]:


encoding = pd.DataFrame(encoding,columns = ohe_columns)


# In[ ]:


encoding


# In[ ]:


encoding = encoding.reset_index(drop=True)
sample_data =sample_data.reset_index(drop=True)
sample_data=pd.concat([encoding,sample_data],axis=1)


# In[ ]:


pd.isna(sample_data).sum()


# In[ ]:


sample_data


# In[ ]:


x_test_notnull


# #x_test에 대해 occpuation 결측치 채워넣기

# In[ ]:


pd.isna(x_test).sum()


# #occupation ? 분리

# In[ ]:


x_test_null = x_test[x_test['occupation']=='?']
x_test_notnull = x_test[x_test['occupation'] !='?']


# In[ ]:


workclass_null = x_test_null['workclass']
occupation_null = x_test_null['occupation']
id_null = x_test_null['id']

x_test_null.drop(columns=['workclass','occupation','id'],inplace=True)


# In[ ]:


x_test_null


# #predict

# In[ ]:


predict = xgb.predict(x_test_null)


# In[ ]:


col = ['occupation']


# In[ ]:


occupation = pd.DataFrame(predict,columns=col)


# In[ ]:


occupation


# In[ ]:


x_test_notnull


# In[ ]:


x_test_null = x_test_null.reset_index(drop=True)
workclass_null = workclass_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,workclass_null],axis=1)


# In[ ]:


x_test_null = x_test_null.reset_index(drop=True)
occupation = occupation.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,occupation],axis=1)


# In[ ]:


x_test_null = x_test_null.reset_index(drop=True)
id_null = id_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,id_null],axis=1)


# In[ ]:


x_test_null= x_test_null.reset_index(drop=True)
x_test_notnull = x_test_notnull.reset_index(drop=True)
x_test = pd.concat([x_test_null,x_test_notnull],axis=0)


# In[ ]:


x_test


# #workplace occupation 에 대해 인코딩 실시

# In[ ]:


encoding = x_test[['workclass','occupation']]
x_test.drop(columns=['workclass','occupation'],inplace=True)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
encoding = ohe.fit_transform(encoding)


# In[ ]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[ ]:


encoding = pd.DataFrame(encoding,columns = ohe_columns)


# In[ ]:


encoding = encoding.reset_index(drop=True)
x_test =x_test.reset_index(drop=True)
x_test=pd.concat([encoding,x_test],axis=1)


# In[ ]:


x_test.columns


# In[ ]:


sample_data.columns


# In[ ]:


num_columns


# #전처리 완료

# In[ ]:


x_test.sort_values(by='id',inplace=True)


# In[ ]:


x_test.drop(columns='id',inplace=True)


# In[ ]:


x_test['Never-worked'] = 0


# In[ ]:


label = sample_data['income']


# In[ ]:


sample_data.drop(columns='income',inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(label)


# In[ ]:


sample_data.columns


# In[ ]:


x_test = x_test[['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
       'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Adm-clerical',
       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',
       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
       'Tech-support', 'Transport-moving', 'fnlwgt', 'hours_per_week',
       'capital', 'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed',
       'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
       'Wife', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other',
       'White', 'Female', 'Male', 'elder', 'middle', 'middle_elder', 'youth',
       'Africa', 'ES_Asia', 'E_Asia', 'High_Europe', 'Mid_Europe', 'N_America',
       'S_America', 'United-States', 'Used_Colony', 'W_Asia', 'assoc',
       'bachelors', 'college', 'elementry', 'highschool', 'highschool_grad',
       'middleschool', 'ms', 'ph.d', 'pre']]


# In[ ]:


x_test


# In[ ]:


sample_data


# In[ ]:


sample_data[num_columns]


# #test(only split)

# In[ ]:


sample_data


# In[ ]:


x_test


# 간단한 모델

# In[ ]:


from sklearn.model_selection import train_test_split

tmp_train, tmp_valid, y_train, y_valid = train_test_split(sample_data, label, 
                                                          test_size=0.3,
                                                          random_state=2020,
                                                          shuffle=True,
                                                          stratify=label)


# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
lgb = LGBMClassifier(tree_method='gpu_hist')

lgb.fit(tmp_train, y_train)

y_pred = lgb.predict(tmp_valid)

print(f"LightGBM F1 Score: {f1_score(y_valid, y_pred, average='micro')}")


# OOF ENSEMBLE 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




