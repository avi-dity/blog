import streamlit as st
from PIL import Image


st.sidebar.title("Enough")
pages = [
    "Introduction","Preprocessing","Sampling","Models","Performance Metrics","Final code"
]


page = st.sidebar.radio("Go to", pages)

if page == "Introduction":
    image = Image.open("mltree.png")
    st.image(image, use_column_width=True)
elif page=="Performance Metrics":
    st.header('Regression')
    
    st.write("<h5>Mean Absolute Error (MAE)<h5>",unsafe_allow_html=True)
    st.latex(r"\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|")
    st.code("""from sklearn.metrics import mean_absolute_error""", language='python')
    
    st.write("<h5>Mean Squared Error (MSE)<h5>",unsafe_allow_html=True)
    st.latex(r"\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2")
    st.code("""from sklearn.metrics import mean_squared_error""", language='python')
    
    st.write("<h5>Root Mean Squared Error (RMSE)<h5>",unsafe_allow_html=True)
    st.latex(r"\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}")
    st.code("""from sklearn.metrics import mean_squared_error\nimport numpy as np\nnp.sqrt(mean_squared_error(y_test,y_pred))""", language='python')

    st.write("<h5>R-Squared<h5>",unsafe_allow_html=True)
    st.latex(r"1-\frac{SS_{RES}}{SS_{TOT}}=1-\frac{\sum_i(y-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}")
    st.code("""from sklearn.metrics import r2_score""", language='python')
    
    st.write("<h5>Adjusted R-squared<h5>",unsafe_allow_html=True)
    st.latex(r"1-\frac{(1-R^2)(N-1)}{N-p-1}")
    st.code("""from sklearn.metrics import r2_score\nimport numpy as np\nr2=r2_score(y_test,y_pred)\nn=len(y_test)\n#Number of features\nk=1\nadjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)""", language='python')

    st.header('Classification')

    st.write("<h5>Accuracy<h5>",unsafe_allow_html=True)
    st.latex(r"\frac{TP+TN}{TP+FP+FN+TN}")
    st.code("""from sklearn.metrics import accuracy_score""", language='python')

    st.write("<h5>Precision and Recall<h5>",unsafe_allow_html=True)
    st.latex(r"Precision=\frac{TP}{TP+FP} \ \ \ \ \ Recall=\frac{TP}{TP+FN}")
    st.code("""from sklearn.metrics import precision_score, recall_score""", language='python')

    st.write("<h5>Specificity<h5>",unsafe_allow_html=True)
    st.latex(r"\frac{TN}{TN+FP}")
    st.code("""from sklearn.metrics import confusion_matrix\ntn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()\nspecificity = tn / (tn + fp)""", language='python')

    st.write("<h5>F1-score<h5>",unsafe_allow_html=True)
    st.latex(r"2*\frac{precision*recall}{precision+recall}")
    st.code("""from sklearn.metrics import f1_score""", language='python')

    st.write("<h5>AUC-ROC<h5>",unsafe_allow_html=True)
    st.latex(r"True \ Positive \ Rate(TPR) = \frac{TP}{TP+FN}")
    st.latex(r"False \ Positive Rate(FPR) = \frac{FP}{FP+TN}")
    st.code("""from sklearn.metrics import roc_auc_score""", language='python')

elif page=="Preprocessing":
    st.header('Encoding')
    
    st.write("<h5>Label Encoder<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.preprocessing import LabelEncoder""", language='python')
    
    st.write("<h5>One-Hot Encoding<h5>",unsafe_allow_html=True)
    st.code("""import pandas as pd\npd.get_dummies(df['column'])""", language='python')
    
    st.header('Scaling')
    
    st.write("<h5>Standard Scaler<h5>",unsafe_allow_html=True)
    st.latex(r"y=\frac{x-mean}{Standard Deviation}")
    st.code("""from sklearn.preprocessing import StandardScaler""", language='python')

    st.write("<h5>Min-Max Scaler<h5>",unsafe_allow_html=True)
    st.latex(r"y=\frac{x-min}{max-min}")
    st.code("""from sklearn.preprocessing import MinMaxScaler""", language='python')

elif page=="Sampling":
    st.header('Hold out')
    hold = Image.open("hold_one.jpeg")
    st.image(hold, use_column_width=True)
    st.code("""from sklearn.model_selection import train_test_split""", language='python')

    st.header('Leave One Out')
    leave = Image.open("leave.png")
    st.image(leave, use_column_width=True)
    st.code("""from sklearn.model_selection import LeaveOneOut""", language='python')

    st.header('K-Fold')
    kfold = Image.open("k-fold.png")
    st.image(kfold, use_column_width=True)
    st.code("""from sklearn.model_selection import KFold""", language='python')

    st.header('Stratified K-Fold')
    skfold= Image.open("stratified.png")
    st.image(skfold, use_column_width=True)
    st.code("""from sklearn.model_selection import StratifiedKFold""", language='python')

elif page=="Models":
    st.header("Regression")
    
    st.write("<h5>Linear Regression<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.linear_model import LinearRegression""", language='python')
    
    st.write("<h5>Decision Tree<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.tree import DecisionTreeRegressor""", language='python')
    
    st.write("<h5>LassoCV<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.linear_model import LassoCV""", language='python')

    st.write("<h5>Support Vector Machines<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.svm import SVR""", language='python')

    st.header("Classification")

    st.write("<h5>Logistic Regression<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.linear_model import LogisticRegression""", language='python')
    
    st.write("<h5>Naïve Bayes<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB""", language='python')
    
    st.write("<h5>Decision Tree<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.tree import DecisionTreeClassifier""", language='python')

    st.write("<h5>Support Vector Machines<h5>",unsafe_allow_html=True)
    st.code("""from sklearn.svm import SVC""", language='python')
elif page=="Final code":
    st.header("Regression")
    regression="""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score ,mean_squared_error ,mean_absolute_error
from sklearn import datasets

x = pd.DataFrame(datasets.load_diabetes().data,columns=datasets.load_diabetes().feature_names)
y=pd.DataFrame(datasets.load_diabetes().target,columns=['diabetes'])

scale=MinMaxScaler()
label=OrdinalEncoder()
x[x.select_dtypes(exclude='object').columns]=scale.fit_transform(x.select_dtypes(exclude='object'))
x[x.select_dtypes(include='object').columns]=label.fit_transform(x.select_dtypes(include='object'))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(f'''Accuracy: {r2_score(y_test,y_pred)}
MSE: {mean_squared_error(y_test,y_pred)}
MAE: {mean_absolute_error(y_test,y_pred)}''')
"""
    st.code(regression, language='python')
    st.header("Classification")
else:
    st.header(page)
    st.write("Content for " + page + " will be added here.")

 
