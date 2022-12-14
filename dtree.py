import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Train your own Decision Tree model')

df = pd.read_csv('diabetes.csv')
st.dataframe(df)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

split23 = st.sidebar.slider('Choose the test size', 1, 99, 10)
splittrain = 100 - split23
split2 = split23/100
rd = st.sidebar.slider('Choose the train test split random state', 0, 42, 0)

st.sidebar.write("**Decision Tree Parameters**")

rd2 = st.sidebar.slider('Choose the Decision Tree random state', 0, 42, 0)
max_d = st.sidebar.slider('Choose the Decision Tree Maximum Depth', 3, 7, 7)
crit = st.sidebar.selectbox("Choose your Decision Tree Criterion :",('gini', 'entropy'))
split = st.sidebar.selectbox("Choose your Decision Tree Splitter :",('best', 'random'))

st.write("Your train size : ", splittrain)
st.write("Your test size : ", split23)
st.write("Your train test split random state : ", rd)
st.write("Your Decision Tree random state : ", rd2)
st.write("Your Decision Tree Maximum Depth : ", max_d)
st.write("Your Decision Tree Criterion : ", crit)
st.write("Your Decision Tree Splitter : ", split)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split2, random_state=rd)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state = rd2, max_depth=max_d, criterion=crit, splitter=split)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
acc = round(accuracy_score(y_test, y_pred)*100 ,2)
y_pred = dtree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
figure = plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

f12 = round(f1*100,2)
prec2 = round(prec*100,2)
recall2 = round(recall*100,2)

from sklearn import tree
from sklearn.tree import plot_tree
import graphviz 
dot_data = tree.export_graphviz(dtree, out_file=None) 
graph = graphviz.Source(dot_data) 
dot_data = tree.export_graphviz(dtree, out_file=None,     
filled=True, rounded=True,  
special_characters=True)  
graph = graphviz.Source(dot_data) 


st.write("**Algorithm Accuracy in (%)**")
st.info(acc)
st.write("**Precision (%)**")
st.info(prec2)
st.write("**Recall (%)**")
st.info(recall2)
st.write("**F-1 Score (%)**")
st.info(f12)
st.write("**Confusion Matrix (%)**")
st.write(figure)
st.write("**Feature Importance**")
st.bar_chart(dtree.feature_importances_)
st.write("**Decision Tree**")
st.graphviz_chart(dot_data)

