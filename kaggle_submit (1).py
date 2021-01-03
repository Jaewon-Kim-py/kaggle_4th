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

# In[3]:


#파생변수 생성(capital)
sample_data['capital'] = sample_data['capital_gain'] - sample_data['capital_loss']


# In[4]:


sample_data.drop(columns=['capital_gain','capital_loss'],inplace=True)


# In[5]:


#파생변수 생성(age_group)
bins = [0, 29, 39, 64, 120]
labels = ['youth', 'middle', 'middle_elder', 'elder']
sample_data['age_group'] = pd.cut(sample_data.age, bins, labels = labels,include_lowest = True)


# In[6]:


sample_data.drop(columns=['age'],inplace=True)


# In[7]:


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


# In[8]:


sample_data.drop(columns=['native_country'], inplace=True)


# In[9]:


#education 변수
sample_data['education'].unique()


# In[10]:


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


# In[11]:


sample_data.drop(columns=['education','education_num'],inplace=True)


# 데이터 확인

# In[12]:


sample_data


# #미리 제거하거나 분리해야할 변수 설정
# (income의 경우 target, id 경우 필요없는 변수, workclass,occupation의 경우 따로 예측할
# 변수이므로)

# In[13]:


label = sample_data['income']
workclass = sample_data['workclass']
occupation = sample_data['occupation']


# In[14]:


sample_data.drop(columns=['id','workclass','occupation','income'], inplace=True)


# 연속형, 범주형 변수 구분

# In[15]:


cat_columns = [c for (c, t) in zip(sample_data.dtypes.index, sample_data.dtypes) if t == 'O'] 
num_columns = [c for c in sample_data.columns if c not in cat_columns]


# In[16]:


sample_data['age_group'] = sample_data['age_group'].astype('object')


# In[17]:


cat_columns = [c for (c, t) in zip(sample_data.dtypes.index, sample_data.dtypes) if t == 'O'] 
num_columns = [c for c in sample_data.columns if c not in cat_columns]


# In[18]:


sample_data_num = sample_data[num_columns]


# In[19]:


sample_data_cat = sample_data[cat_columns]


# 범주형 변수의 one-hot encoding 실시

# In[20]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
sample_data_cat = ohe.fit_transform(sample_data_cat)


# In[21]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[22]:


sample_data_cat = pd.DataFrame(sample_data_cat,columns = ohe_columns)


# 연속형, 범주형 분리한 것 합치기
# income, occupation 붙이기(결측치에 대해 예측하여야 하므로 따로 분리 후 합침)

# In[23]:


sample_data_num.reset_index(drop=True)
sample_data_cat.reset_index(drop=True)
sample_data1 = pd.concat([sample_data_num,sample_data_cat],axis=1)


# In[24]:


sample_data1.reset_index(drop=True)
label = label.reset_index(drop=True)
sample_data1 = pd.concat([sample_data1,label],axis=1)


# In[25]:


sample_data1.reset_index(drop=True)
occupation = occupation.reset_index(drop=True)
sample_data1 = pd.concat([sample_data1,occupation],axis=1)


# In[26]:


sample_data1


# workclass 붙이기(occupation과 같은 이유)

# In[27]:


workclass.reset_index(drop=True)
sample_data1.reset_index(drop=True)
sample_data_workclass = pd.concat([sample_data1,workclass],axis=1)


# In[28]:


sample_data_workclass


# 예측 및 타겟변수 분리 실시

# In[29]:


sample_data_null = sample_data_workclass[sample_data_workclass['workclass']=='?']
sample_data_notnull = sample_data_workclass[sample_data_workclass['workclass']!='?']


# In[30]:


null_income = sample_data_null['income']
notnull_income = sample_data_notnull['income']
sample_data_null.drop(columns=['income'],inplace=True)
sample_data_notnull.drop(columns=['income'],inplace=True)


# In[31]:


null_occupation = sample_data_null['occupation']
notnull_occupation = sample_data_notnull['occupation']
sample_data_null.drop(columns=['occupation'],inplace=True)
sample_data_notnull.drop(columns=['occupation'],inplace=True)


# In[32]:


target1 = sample_data_notnull['workclass']
sample_data_null.drop(columns=['workclass'],inplace=True)
sample_data_notnull.drop(columns=['workclass'],inplace=True)


# workclass='?' 에 대한 예측 모델 생성

# In[33]:


sample_data_null1 = sample_data_null.copy()


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(sample_data_notnull, target1, 
                                                      test_size=0.3,
                                                      random_state=2020,
                                                      shuffle=True)


# In[35]:


x_train_mean = np.mean(x_train[num_columns], axis=0)
x_train_std  = np.std(x_train[num_columns], axis=0)


x_train.loc[:, num_columns] = (x_train[num_columns] - x_train_mean) / x_train_std 
x_valid.loc[:, num_columns] = (x_valid[num_columns] - x_train_mean) / x_train_std
sample_data_null1.loc[:, num_columns] = (sample_data_null1[num_columns] - x_train_mean) / x_train_std


# In[36]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[37]:


xgb.fit(x_train,y_train)


# In[38]:


workclass_pred = xgb.predict(sample_data_null1)


# In[39]:


col = ['workclass']


# In[40]:


workclass_pred = pd.DataFrame(workclass_pred,columns=col)


# 합치기(1)- sample_data_null 과 workclass_pred

# In[41]:


workclass_pred = workclass_pred.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, workclass_pred], axis=1)


# In[42]:


sample_data_null = sample_data_null.reset_index(drop=True)
null_income = null_income.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, null_income], axis=1)


# In[43]:


sample_data_null = sample_data_null.reset_index(drop=True)
null_occupation = null_occupation.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,null_occupation],axis=1)


# In[44]:


sample_data_null


# 합치기(2) sample_data_notnull

# In[45]:


sample_data_notnull.reset_index(drop=True)
target1.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,target1],axis=1)


# In[46]:


sample_data_notnull


# income 합치기

# In[47]:


sample_data_notnull.reset_index(drop=True)
notnull_income.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,notnull_income],axis=1)


# In[48]:


sample_data_notnull


# occupation 합치기

# In[49]:


sample_data_notnull.reset_index(drop=True)
notnull_occupation.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,notnull_occupation],axis=1)


# In[50]:


sample_data_notnull


# 합치기

# In[51]:


sample_data_notnull.reset_index(drop=True)
sample_data_null.reset_index(drop=True)
sample_data = pd.concat([sample_data_notnull,sample_data_null],axis=0)


# In[52]:


sample_data


# test.csv에 대한 전처리

# In[53]:


x_test = pd.read_csv(join('data', join('kaggle', 'test.csv')))


# In[54]:


x_test['capital'] = x_test['capital_gain'] - x_test['capital_loss']


# In[55]:


x_test.drop(columns=['capital_gain','capital_loss'],inplace=True)


# In[56]:


#파생변수 생성(age_group)
bins = [0, 29, 39, 64, 120]
labels = ['youth', 'middle', 'middle_elder', 'elder']
x_test['age_group'] = pd.cut(x_test.age, bins, labels = labels,include_lowest = True)


# In[57]:


x_test.drop(columns=['age'],inplace=True)


# In[58]:


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


# In[59]:


x_test.drop(columns=['native_country'], inplace=True)


# In[60]:


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


# In[61]:


x_test.drop(columns=['education','education_num'], inplace=True)


# In[62]:


x_test


# In[63]:


#변수 분리


# In[64]:


workclass=x_test['workclass']
occupation = x_test['occupation']
id = x_test['id']
x_test.drop(columns=['workclass','occupation','id'],inplace=True)


# In[65]:


#연속형 범주형 구분


# In[66]:


cat_columns = [c for (c, t) in zip(x_test.dtypes.index, x_test.dtypes) if t == 'O'] 
num_columns = [c for c in x_test.columns if c not in cat_columns]


# In[67]:


x_test['age_group'] = x_test['age_group'].astype('object')


# In[68]:


cat_columns = [c for (c, t) in zip(x_test.dtypes.index, x_test.dtypes) if t == 'O'] 
num_columns = [c for c in x_test.columns if c not in cat_columns]


# In[69]:


x_test_num = x_test[num_columns]
x_test_cat = x_test[cat_columns]


# In[70]:


#범주형 인코딩


# In[71]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
x_test_cat = ohe.fit_transform(x_test_cat)


# In[72]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[73]:


x_test_cat = pd.DataFrame(x_test_cat,columns = ohe_columns)


# In[74]:


#num col 합치기


# In[75]:


x_test_num.reset_index(drop=True)
x_test_cat.reset_index(drop=True)
x_test = pd.concat([x_test_num,x_test_cat],axis=1)


# In[76]:


x_test


# In[77]:


#workclass ocupation 붙이기


# In[78]:


x_test = x_test.reset_index(drop=True)
workclass= workclass.reset_index(drop=True)
x_test = pd.concat([x_test,workclass],axis=1)


# In[79]:


x_test.reset_index(drop=True)
occupation.reset_index(drop=True)
x_test = pd.concat([x_test,occupation],axis=1)


# In[80]:


x_test.reset_index(drop=True)
id.reset_index(drop=True)
x_test = pd.concat([x_test,id],axis=1)


# In[81]:


# original data
x_test


# In[82]:


#occupation 분리


# In[83]:


x_test_null = x_test[x_test['occupation']=='?']
x_test_notnull = x_test[x_test['occupation'] !='?']


# In[84]:


x_test_null


# In[85]:


x_test_notnull


# In[86]:


workclass_null = x_test_null['workclass']
occupation_null = x_test_null['occupation']
id_null =x_test_null['id']
x_test_null.drop(columns=['workclass','occupation','id'],inplace=True)


# In[87]:


#test 내용이 train에 섞이지 않도록 주의
predict = xgb.predict(x_test_null)


# In[88]:


col = ['workclass']


# In[89]:


predict = pd.DataFrame(predict,columns=col)


# In[90]:


x_test_null = x_test_null.reset_index(drop=True)
predict = predict.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,predict],axis=1)


# In[91]:


x_test_null


# In[92]:


#occupation 붙이기


# In[93]:


occupation_null = occupation_null.reset_index(drop=True)
x_test_null = x_test_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null, occupation_null],axis=1)


# In[94]:


x_test_null


# In[95]:


#id 붙이기


# In[96]:


id_null = id_null.reset_index(drop=True)
x_test_null = x_test_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,id_null],axis=1)


# In[97]:


#null notnull 합치기


# In[98]:


x_test_null = x_test_null.reset_index(drop=True)
x_test_notnull = x_test_notnull.reset_index(drop=True)
x_test = pd.concat([x_test_null,x_test_notnull],axis=0)


# In[99]:


x_test


# occupation 모델 만들기

# In[100]:


sample_data


# #occupation ?  구분

# In[101]:


sample_data_null = sample_data[sample_data['occupation']=='?']
sample_data_notnull = sample_data[sample_data['occupation'] !='?']


# In[102]:


sample_data_null


# #변수 분리 및 제거

# In[103]:


workclass_null = sample_data_null['workclass']
income_null = sample_data_null['income']
sample_data_null.drop(columns=['workclass','income','occupation'],inplace=True)


# In[104]:


workclass_notnull = sample_data_notnull['workclass']
income_notnull = sample_data_notnull['income']
target = sample_data_notnull['occupation']
sample_data_notnull.drop(columns=['workclass','income','occupation'],inplace=True)


# #occupation 예측 모델 생성

# In[105]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(sample_data_notnull, target, 
                                                      test_size=0.3,
                                                      random_state=2020,
                                                      shuffle=True)


# In[106]:


sample_data_null1 = sample_data_null.copy()


# In[107]:


x_train_mean = np.mean(x_train[num_columns], axis=0)
x_train_std  = np.std(x_train[num_columns], axis=0)


x_train.loc[:, num_columns] = (x_train[num_columns] - x_train_mean) / x_train_std 
x_valid.loc[:, num_columns] = (x_valid[num_columns] - x_train_mean) / x_train_std
sample_data_null1.loc[:, num_columns] = (sample_data_null1[num_columns] - x_train_mean) / x_train_std


# In[108]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[109]:


xgb.fit(x_train,y_train)


# In[110]:


occupation_pred = xgb.predict(sample_data_null1)


# In[111]:


col = ['occupation']


# In[112]:


occupation_pred = pd.DataFrame(occupation_pred,columns=col)


# #workclass 합치기

# In[113]:


sample_data_null = sample_data_null.reset_index(drop=True)
workclass_null = workclass_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,workclass_null],axis=1)


# #합치기(1)- sample_data_null 과 occupation_pred

# In[114]:


occupation_pred = occupation_pred.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null,occupation_pred],axis=1)


# In[115]:


sample_data_null


# #income 합치기

# In[116]:


sample_data_null = sample_data_null.reset_index(drop=True)
income_null = income_null.reset_index(drop=True)
sample_data_null = pd.concat([sample_data_null, income_null], axis=1)


# In[117]:


sample_data_null


# sample_data_notnull 붙이기

# In[118]:


sample_data_notnull.reset_index(drop=True)
workclass_notnull.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,workclass_notnull],axis=1)


# In[119]:


sample_data_notnull.reset_index(drop=True)
target.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,target],axis=1)


# In[120]:


sample_data_notnull.reset_index(drop=True)
income_notnull.reset_index(drop=True)
sample_data_notnull = pd.concat([sample_data_notnull,income_notnull],axis=1)


# #sample_data_notnull sample_data_null 합치기

# In[121]:


sample_data_notnull = sample_data_notnull.reset_index(drop=True)
sample_data_null = sample_data_null.reset_index(drop=True)
sample_data = pd.concat([sample_data_null,sample_data_notnull],axis=0)


# In[122]:


sample_data


# #workclass occupation 인코딩 처리

# In[123]:


encoding = sample_data[['workclass','occupation']]
sample_data.drop(columns=['workclass','occupation'],inplace=True)


# In[124]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
encoding = ohe.fit_transform(encoding)


# In[125]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[126]:


encoding = pd.DataFrame(encoding,columns = ohe_columns)


# In[127]:


encoding


# In[128]:


encoding = encoding.reset_index(drop=True)
sample_data =sample_data.reset_index(drop=True)
sample_data=pd.concat([encoding,sample_data],axis=1)


# In[129]:


pd.isna(sample_data).sum()


# In[130]:


sample_data


# In[131]:


x_test_notnull


# #x_test에 대해 occpuation 결측치 채워넣기

# In[132]:


pd.isna(x_test).sum()


# #occupation ? 분리

# In[133]:


x_test_null = x_test[x_test['occupation']=='?']
x_test_notnull = x_test[x_test['occupation'] !='?']


# In[134]:


workclass_null = x_test_null['workclass']
occupation_null = x_test_null['occupation']
id_null = x_test_null['id']

x_test_null.drop(columns=['workclass','occupation','id'],inplace=True)


# In[135]:


x_test_null


# #predict

# In[136]:


predict = xgb.predict(x_test_null)


# In[137]:


col = ['occupation']


# In[138]:


occupation = pd.DataFrame(predict,columns=col)


# In[139]:


occupation


# In[140]:


x_test_notnull


# In[141]:


x_test_null = x_test_null.reset_index(drop=True)
workclass_null = workclass_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,workclass_null],axis=1)


# In[142]:


x_test_null = x_test_null.reset_index(drop=True)
occupation = occupation.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,occupation],axis=1)


# In[143]:


x_test_null = x_test_null.reset_index(drop=True)
id_null = id_null.reset_index(drop=True)
x_test_null = pd.concat([x_test_null,id_null],axis=1)


# In[144]:


x_test_null= x_test_null.reset_index(drop=True)
x_test_notnull = x_test_notnull.reset_index(drop=True)
x_test = pd.concat([x_test_null,x_test_notnull],axis=0)


# In[145]:


x_test


# #workplace occupation 에 대해 인코딩 실시

# In[146]:


encoding = x_test[['workclass','occupation']]
x_test.drop(columns=['workclass','occupation'],inplace=True)


# In[147]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
encoding = ohe.fit_transform(encoding)


# In[148]:


ohe_columns=ohe.categories_[0].tolist()
for i in range(len(ohe.categories_)-1):
        ohe_columns += ohe.categories_[i+1].tolist()


# In[149]:


encoding = pd.DataFrame(encoding,columns = ohe_columns)


# In[150]:


encoding = encoding.reset_index(drop=True)
x_test =x_test.reset_index(drop=True)
x_test=pd.concat([encoding,x_test],axis=1)


# In[151]:


x_test.columns


# In[152]:


sample_data.columns


# In[153]:


num_columns


# #전처리 완료

# In[154]:


x_test.sort_values(by='id',inplace=True)


# In[155]:


x_test.drop(columns='id',inplace=True)


# In[156]:


x_test['Never-worked'] = 0


# In[157]:


label = sample_data['income']


# In[158]:


sample_data.drop(columns='income',inplace=True)


# In[159]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(label)


# In[160]:


sample_data.columns


# In[161]:


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


# In[162]:


x_test


# In[163]:


sample_data


# In[164]:


sample_data[num_columns]


# #test(only split)

# In[165]:


sample_data


# In[166]:


x_test


# 간단한 모델

# In[167]:


from sklearn.model_selection import train_test_split

tmp_train, tmp_valid, y_train, y_valid = train_test_split(sample_data, label, 
                                                          test_size=0.3,
                                                          random_state=2020,
                                                          shuffle=True,
                                                          stratify=label)


# In[168]:


from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
lgb = LGBMClassifier(tree_method='gpu_hist')

lgb.fit(tmp_train, y_train)

y_pred = lgb.predict(tmp_valid)

print(f"LightGBM F1 Score: {f1_score(y_valid, y_pred, average='micro')}")


# OOF ENSEMBLE 

# 전처리 프로세스(정규화)

# In[171]:


def preprocess(x_train,x_valid,x_test):
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()
    tmp_x_test = x_test.copy()
    
    
    
        
    x_train_mean = np.mean(tmp_x_train[num_columns], axis=0)
    x_train_std  = np.std(tmp_x_train[num_columns], axis=0)

    tmp_x_train.loc[:, num_columns] = (tmp_x_train[num_columns] - x_train_mean) / (x_train_std + 1e-4)
    tmp_x_valid.loc[:, num_columns] = (tmp_x_valid[num_columns] - x_train_mean) / (x_train_std + 1e-4)
    tmp_x_test.loc[:, num_columns] = (tmp_x_test[num_columns] - x_train_mean) / (x_train_std + 1e-4)
    
    
    
    return tmp_x_train, tmp_x_valid, tmp_x_test


# StratifiedKfold 실시

# In[175]:


from sklearn.model_selection import StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)


# In[176]:


#stackoverflow 참고
def xgb_f1(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int) 
    return 'f1', f1_score(t, y_bin, average='micro')


# In[177]:


val_scores = list()
oof_pred = np.zeros((x_test.shape[0], ))

for i, (trn_idx, val_idx) in enumerate(skf.split(sample_data, label)):
    x_train, y_train = sample_data.iloc[trn_idx, :], label[trn_idx]
    x_valid, y_valid = sample_data.iloc[val_idx, :], label[val_idx]
    
    # 전처리
    x_train, x_valid, x_test = preprocess(x_train, x_valid, x_test)
    
    
    
    # 모델생성
    clf = XGBClassifier(n_estimators=100,
                        random_state=2020,
                        tree_method='gpu_hist',
                        n_jobs=2)
    
    # 모델학습
    clf.fit(x_train, y_train,
            eval_set = [[x_valid, y_valid]], 
            eval_metric = xgb_f1,        
            early_stopping_rounds = 100,
            verbose = 100,  )

    # 훈련, 검증 데이터 F1 Score 확인
    trn_f1_score = f1_score(y_train, clf.predict(x_train))
    val_f1_score = f1_score(y_valid, clf.predict(x_valid))
    print('{} Fold, train f1_score : {:.4f}4, validation f1_score : {:.4f}\n'.format(i, trn_f1_score, val_f1_score))
    
    val_scores.append(val_f1_score)
    
    oof_pred += clf.predict_proba(x_test)[: , 1] / n_splits
    

# 교차 검증 F1 Score 평균 계산
print('Cross Validation Score : {:.4f}'.format(np.mean(val_scores)))


# In[184]:


(oof_pred>0.5).astype(int)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




