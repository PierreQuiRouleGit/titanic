import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time

import streamlit as st

from urllib.request import urlopen

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pickle

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope

import mlflow
from pyngrok import ngrok

import shap
import lime
from lime import lime_tabular


#data characteristics
@st.cache_data
def load_df():
    titanic_test = pd.read_csv("titanic_test_nettoyer.csv")
    scaler = StandardScaler()
    scaler.fit(titanic_test)
    scaled_test = scaler.transform(titanic_test)
    df_test = pd.DataFrame(scaled_test, index=titanic_test.index, columns=titanic_test.columns)
    drop_col = ['Unnamed: 0','PassengerId']
    df_test.drop(columns=drop_col,inplace=True)
    return df_test

titanic_test = load_df()

@st.cache_data
def load_df_test_details():
    df_test = pd.read_csv("titanic_test_details.csv")
    df_test.drop(columns=['Unnamed: 0'],inplace=True)
    return df_test

df_test = load_df_test_details()


#best model
@st.cache_resource
def load_model():
    pickle_model = open('model.pkl', 'rb') 
    clf = pickle.load(pickle_model)
    return clf

clf = load_model()

list_id = []
for i in range(len(df_test)):
    list_id.append(df_test['PassengerId'][i])


#id
input_id = st.number_input('Write PassengerId between 892 and 1309',format="%i")

if input_id in list_id:
    st.subheader('Taux de survie (0 bad - 1 good)')
    proba = clf.predict_proba(titanic_test)

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(proba[df_test.loc[df_test['PassengerId']==int(input_id)].index.item()][1],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = { 'axis': {'range': [0, 1]},
            'bar' :{'color': "black"},
            'steps' : [
                {'range': [0, 0.33], 'color': "green"},
                {'range': [0.33, 0.66], 'color': "yellow"},
                {'range': [0.66, 1], 'color': "red"}]
        
        },
        title = {'text': "Score"}))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Status')

    st.subheader('Feature importances')
    feature_importance = (
                    pd.DataFrame(
                    {
                        'variable': titanic_test.columns,
                        'coefficient' : clf.coef_[0]
                    }
                    )
                    .round(decimals=2)  \
                    .sort_values('coefficient',ascending=False)  \
                    .style.bar(color=['red','green'],align='zero')
                    )
    st.write(feature_importance)

    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(titanic_test),feature_names=titanic_test.columns,class_names=['target'],mode='regression')
    exp = explainer.explain_instance(data_row=titanic_test.iloc[df_test.loc[df_test['PassengerId']==int(input_id)].index.item()], predict_fn=clf.predict_proba)
        
    st.set_option('deprecation.showPyplotGlobalUse', False)
    exp.as_pyplot_figure()
    st.pyplot()
            

    st.write("Gender : ", df_test.loc[df_test['PassengerId']==int(input_id),'Sex'].values[0])
    st.write("Age :", df_test.loc[df_test['PassengerId']==int(input_id),'Age'].values[0])
    st.write("Embarked :", df_test.loc[df_test['PassengerId']==int(input_id),'Embarked'].values[0])
    st.write("Fare :", df_test.loc[df_test['PassengerId']==int(input_id),'Fare'].values[0])
    st.write("Pclass :", df_test.loc[df_test['PassengerId']==int(input_id),'Pclass'].values[0])
    st.write("Family_Size :", df_test.loc[df_test['PassengerId']==int(input_id),'Family_Size'].values[0])
    st.write("SibSp :", df_test.loc[df_test['PassengerId']==int(input_id),'SibSp'].values[0])
    st.write("Parch :", df_test.loc[df_test['PassengerId']==int(input_id),'Parch'].values[0])
    st.write("Family_Size_Grouped  :", df_test.loc[df_test['PassengerId']==int(input_id),'Family_Size_Grouped'].values[0])
    st.write("Title :", df_test.loc[df_test['PassengerId']==int(input_id),'Title'].values[0])
    st.write("Is_Married:", df_test.loc[df_test['PassengerId']==int(input_id),'Is_Married'].values[0])
else:
    st.write('Waiting for ID or Wrong ID')

st.subheader("----"*20)
st.title('Can you survive ? Titanic')
gender_col =['male','female']

gender = st.selectbox(
        'Select genre ?',
        options = gender_col)
        
st.write('You selected:', gender)

age_col = ['(0, 5]','(5, 10]','(10, 15]','(15, 20]','(20, 25]','(25, 30]','(30, 35]',
           '(35, 40]','(40, 45]','(45, 50]','(50, 55]','(55, 60]','(65, 70]','(70,75]','(75, 80]']
age = st.selectbox(
        'Select age ?',
        options = age_col)

st.write('You selected:', age)

embarqued_col = ['S','Q','C']

embarqued = st.selectbox(
        'Select embarqued ?',
        options = embarqued_col)

st.write('You selected:', embarqued)

fare_col = ['(-0.001, 7.25]','(7.25, 7.75]','(7.75, 7.896]','(7.896, 8.05]','(8.05, 10.5]',
            '(10.5, 13.0]','(13.0, 15.742]','(15.742, 23.25]','(23.25, 26.55]','(26.55, 34.075]',
            '(34.075, 56.496]','(56.496, 83.475]','(83.475, 512.329]']

fare = st.selectbox(
        'Select fare ?',
        options = fare_col)

st.write('You selected:', fare)

pclass_col = [1,2,3]

pclass = st.selectbox(
        'Select pclass ?',
        options = pclass_col)

st.write('You selected:', pclass)

sibsp_col = [0,1,2,3,4,5,6,7,8]

sibsp = st.selectbox(
        'Select sibsp?',
        options = sibsp_col)

st.write('You selected:', sibsp)

parch_col = [0,1,2,3,4,5,6,7,8,9]

parch = st.selectbox(
        'Select parch?',
        options = parch_col)

st.write('You selected:', parch)

title_col = ['Mr','Miss/Mrs/Ms','Master','Dr/Military/Noble/Clergy']

title = st.selectbox(
        'Select title?',
        options = title_col)

st.write('You selected:', title)

married_col = [0,1]
married = st.selectbox(
        'Select maried?',
        options = married_col)

st.write('You selected:', married)

st.subheader("----"*20)

def survive(age,embarqued,fare,parch,pclass,sex,sibsp,title,married):

    family_size = 1 + parch + sibsp

    if family_size ==1:
        family_size_group = 'Alone'
    elif 2<=family_size <=4:
        family_size_group = 'Small'
    elif 5<=family_size <=6:
        family_size_group = 'Medium'
    else:
        family_size_group = 'Large'
    

    df = pd.DataFrame([[age,embarqued,fare,parch,pclass,sex,sibsp,family_size,family_size_group,title,married]], 
                  columns = ['Age', 'Embarked', 'Fare', 'Parch','Pclass','Sex',
                             'SibSp','Family_Size','Family_Size_Grouped','Title','Is_Married'])
    
    df_concat = pd.concat([df_test, df], sort=True).reset_index(drop=True)
    
    non_numeric_features = ['Embarked', 'Sex', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

    for feature in non_numeric_features:        
        df_concat[feature] = LabelEncoder().fit_transform(df_concat[feature])

    cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_Size_Grouped']
    encoded_features = []
    
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df_concat[feature].values.reshape(-1, 1)).toarray()
        n = df_concat[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df_concat.index
        encoded_features.append(encoded_df)

    df_concat = pd.concat([df_concat, *encoded_features[:5]], axis=1)

    drop_cols = ['Family_Size_Grouped','Title','Embarked','Pclass','Sex']
    df_concat.drop(columns=drop_cols, inplace=True)

    scaler = StandardScaler()
    scaler.fit(df_concat)
    scaled_test = scaler.transform(df_concat)

    df_test_scaled = pd.DataFrame(scaled_test, index=df_concat.index, columns=df_concat.columns)

    df_survive_or_not = df_test_scaled.loc[418:].drop(['PassengerId'], axis=1)

    df_survive_or_not = df_survive_or_not[titanic_test.columns]
       
    return df_survive_or_not





proba_test = clf.predict_proba(survive(age,embarqued,fare,parch,pclass,gender,sibsp,title,married))

fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(proba_test[0,1],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = { 'axis': {'range': [0, 1]},
            'bar' :{'color': "black"},
            'steps' : [
                {'range': [0, 0.33], 'color': "green"},
                {'range': [0.33, 0.66], 'color': "yellow"},
                {'range': [0.66, 1], 'color': "red"}]
        
        },
        title = {'text': "Score"}))
st.plotly_chart(fig, use_container_width=True)
