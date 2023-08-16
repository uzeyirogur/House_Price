import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#data upload
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Missing Values
null_train = train_data.isna().sum().nlargest(81).head(10)
null_test = test_data.isna().sum().nlargest(81).head(10)

train_data = train_data.interpolate()
test_data  = test_data.interpolate()

train_data2 = train_data.copy()
test_data2 = test_data.copy()

train_data2.drop(["Id"],axis=1,inplace=True)
test_data2.drop(["Id"],axis=1,inplace=True)

#her birinden kaçar tane var
value_counts = train_data2["FireplaceQu"].value_counts()

train_data2.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"],axis=1,inplace=True)
test_data2.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"],axis=1,inplace=True)

null_train2 = train_data2.isna().sum().nlargest(81).head(10)
null_test2 = test_data2.isna().sum().nlargest(81).head(10)

train_data2 = train_data2.fillna(train_data2.mean())
test_data2  = test_data2.fillna(train_data2.mean())

null_train3 = train_data2.isna().sum().nlargest(81).head(10)
null_test3 = test_data2.isna().sum().nlargest(81).head(10)

object_features = [x for x in train_data2.columns if train_data2[x].dtype == train_data2['BsmtFinType2'].dtype]

transformers = {
    col: {
        name: i for i, name in enumerate(train_data2[col].unique())
    } for col in object_features
}
train_data2 = train_data2.replace(transformers)
test_data2  = test_data2.replace(transformers)

x = train_data2.drop(["SalePrice"],axis=1)
y = train_data2["SalePrice"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.20, shuffle=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
test_data2 = sc.fit_transform(test_data2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(x_train,y_train)
rfc_predict = rfc.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
import math
r2 = r2_score(y_test, rfc_predict)*100
mse = mean_squared_error(y_test, rfc_predict)
rmse = math.sqrt(mse)

"""
param_gridd = {
    'n_estimators': [10,100, 200, 300,400],            # Ağaç sayısı
}

from sklearn.model_selection import GridSearchCV
gsc = GridSearchCV(estimator=rfc, param_grid=param_gridd,cv=5,verbose=1,scoring="accuracy",n_jobs=-1)

gsc.fit(x_train,y_train)

print(f"En iyi Hyperparametreler : {gsc.best_params_}")
print(f"En iyi score : {gsc.best_score_}")

"""

# OLS
#STATSMODELS OLS
import statsmodels.api as sm
x_train_ols = sm.add_constant(x_train)       #b0 katsayısını hesaplaması için 1 ler ile doldurduk

#Statsmodels ols model oluşturma
sm_model = sm.OLS(y_train,x_train_ols)
sonuc = sm_model.fit()
print(sonuc.summary())

"""
from sklearn.feature_selection import RFE

rfe = RFE(estimator=rfc, n_features_to_select=50)
rfe = rfe.fit(x_train,y_train)
rfe_sonuc = rfe.support_
rfs_secilenler = list(zip(x_train,rfe.support_,rfe.ranking_))

selected_indices = np.where(rfe.support_)[0]  #sadece true olanları döndürür

x_train_selected = x_train[:, selected_indices]

rfc2 = RandomForestClassifier(n_estimators=400)
rfc.rfc.fit(x_train_selected,y_train)
rfc_predict2 = rfc.predict(x_test)


r22 = r2_score(y_test, rfc_predict)*100
mse2 = mean_squared_error(y_test, rfc_predict)
rmse2 = math.sqrt(mse)
"""