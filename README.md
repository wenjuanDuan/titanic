# titanic
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
titanic = pd.read_csv("train.csv")
titanic.head(5)
# titanic.describe()

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# print(titanic.describe())

print(titanic["Sex"].unique())
print(titanic["Embarked"].unique())
print(f"keys:{titanic.keys()}")
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg=RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=2,min_samples_leaf=1)
kf=KFold(n_splits=3,random_state=1,shuffle=True)
golds=model_selection.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
print('randomForest_T10:',golds.mean())

alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
kf=KFold(n_splits=3,random_state=1,shuffle=True)
golds=model_selection.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
print('randomForest_T50:',golds.mean())
