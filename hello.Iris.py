import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st # type: ignore
import joblib as jb

dataset=pd.read_csv(r"C:\Users\prade\OneDrive\Documents\Data Science using Python\Iris.csv")
dataset

x=dataset.drop(["Species","Id"],axis=1)
y=dataset["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

pickle_out=open("classifier.pkl","wb")
pickle.dump(knn,pickle_out)
pickle_out.close()

st.markdown('## Iris Species Prediction')
Sepal_length=st.slider("Sepal Length(cm)")
Sepal_width=st.slider("Sepal Width (cm)")
petal_length=st.slider("petal Length(Cm)")
petal_width=st.slider("Petal Width (Cm)")
if st.button('Predict'):
    model=jb.load("classifier.pkl")
    x=np.array([Sepal_length,Sepal_width,petal_length,petal_width])
    if any(x<=0):
        st.markdown("### Input must be greater than 0")
    else:
        st.markdown(f'### Prediction Is {model.predict([[Sepal_length,Sepal_width,petal_length,petal_width]])}')
        