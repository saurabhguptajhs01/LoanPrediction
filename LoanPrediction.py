import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

import pickle
import streamlit
import plotly.express

def encodeCategorical(df):
    mapGender = {'Male':-1,'Female':1}
    mapMarried = {'Yes':-1,'No':1}
    mapDependents = {'0':0,'1':1,'2':2,'3+':3}
    mapEducation = {'Graduate':-1,'Not Graduate':1}
    mapSelfEmployed = {'Yes':-1,'No':1}
    mapPropertyArea = {'Urban':1,'Rural':2,'Semiurban':3}
    mapLoanStatus = {'Y':1,'N':0}

    df['Gender'] = df['Gender'].map(mapGender)
    df['Married'] = df['Married'].map(mapMarried)
    df['Dependents'] = df['Dependents'].map(mapDependents)
    df['Education'] = df['Education'].map(mapEducation)
    df['Self_Employed'] = df['Self_Employed'].map(mapSelfEmployed)
    df['Property_Area'] = df['Property_Area'].map(mapPropertyArea)
    if 'Loan_Status' in df:
        df['Loan_Status'] = df['Loan_Status'].map(mapLoanStatus)
    return df

def featureEngineering(df):
    df = df.drop('Loan_ID',axis=1)
    return df

def processData(df):
    df = encodeCategorical(df)
    df = featureEngineering(df)
    return df

def create_model():
    df_train = pd.read_csv("data\train.csv")
    df_train = processData(df_train)

    imputer = KNNImputer(missing_values=np.nan,n_neighbors=10)
    imputer.fit_transform(df_train)
    
    numericalCols = df_train.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()
    scaler.fit_transform(df_train[numericalCols])


    y = df_train['Loan_Status']
    X = df_train.drop('Loan_Status',axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    
    preprocessor = make_pipeline(imputer,scaler)

    lr_model = LogisticRegression(max_iter=1000)
    model = make_pipeline(preprocessor,lr_model)
    model.fit(X_train,y_train)
    #y_pred1 = model.predict(X_test)
    #print("accuracy_score",accuracy_score(y_test,y_pred1))

    

    pickle.dump(model,open('model.sav','wb'))
    
    return model

create_model()

def load_model():
    return pickle.load(open('model.sav', 'rb'))

def predict(X):
    model = load_model()
    print("test:",model.predict(X))
    return model.predict_proba(X)

df_test = pd.read_csv("data\test.csv")
df_test = df_test.dropna()
#df_test = processData(df_test)

#predict(df_test)

streamlit.title("Loan Prediction App")
streamlit.sidebar.title("Parameters")

gender = streamlit.sidebar.radio('Select Gender',list(df_test['Gender'].unique()))
married = streamlit.sidebar.radio('Married',list(df_test['Married'].unique()))
dependent = streamlit.sidebar.select_slider('No. of Dependents',[i for i in range(4)])
education = streamlit.sidebar.radio('Education',list(df_test['Education'].unique()))
selfEmployed = streamlit.sidebar.radio('Employed',list(df_test['Self_Employed'].unique()))
applicantIncome = streamlit.sidebar.select_slider('Applicant Income',[i for i in range(df_test['ApplicantIncome'].max())])
coapplicantIncome = streamlit.sidebar.select_slider('Coapplicant Income',[i for i in range(df_test['CoapplicantIncome'].max())])
loanAmount = streamlit.sidebar.select_slider('Loan Amount',[i for i in range(int(df_test['LoanAmount'].max()))])
loanAmountTerm = streamlit.sidebar.select_slider('Loan Term',[i for i in range(600)])
creditHistory = streamlit.sidebar.radio('Credit History',['Yes','No'])
propertyAreaHistory = streamlit.sidebar.radio('Property Area',list(df_test['Property_Area'].unique()))

val = [gender,married,dependent,education,selfEmployed,applicantIncome,coapplicantIncome,loanAmount,loanAmountTerm,
       creditHistory,propertyAreaHistory]

df_val = pd.DataFrame({
    "Loan_ID":[None],
    "Gender":[gender],
    "Married":[married],
    "Dependents":['0'],
    "Education":[education],
    "Self_Employed":[selfEmployed],
    "ApplicantIncome":[applicantIncome],
    "CoapplicantIncome":[coapplicantIncome],
    "LoanAmount":[loanAmount],
    "Loan_Amount_Term":[loanAmountTerm],
    "Credit_History":[1 if creditHistory == 'Yes' else 0],
    "Property_Area":[propertyAreaHistory]
})

df_val = processData(df_val)
#model = load_model()
pred = predict(df_val)[0]
print(pred)

fig3 = plotly.express.pie(["Not accepted", "Accepted"], values=[pred[0], pred[1]],color=["Not accepted", "Accepted"],
color_discrete_map={'Not accepted':'red', 'Accepted':'green'})
fig3.update_layout(
title="<b>Loan approved ?</b>")
streamlit.plotly_chart(fig3)
