import streamlit as st
from PIL import Image
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns

def Train_reg(df,treat,strategy,target,label,scale,model):
    if treat == "Drop":
        for i in df.columns[df.isnull().any()]:
            df.drop(df[df[i].isnull()].index,inplace=True)
    if treat == "Fill":
        cat_df=df.select_dtypes(include=object).columns
        num_df=df.select_dtypes(exclude=object).columns
        num_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df[cat_df]=cat_imputer.fit_transform(df[cat_df])
        df[num_df]=num_imputer.fit_transform(df[num_df])



    x=df.drop(target,axis=1)
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

    trans=make_column_transformer((label,df.select_dtypes(include=object).columns),remainder='passthrough')
    trans.fit_transform(df)
    pipe=Pipeline([('transform',trans),('scale',scale),('clf',model)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)

    return r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred)

def Train_class(df,treat,strategy,target,label,scale,model):
    if treat == "Drop":
        for i in df.columns[df.isnull().any()]:
            df.drop(df[df[i].isnull()].index,inplace=True)
    if treat == "Fill":
        cat_df=df.select_dtypes(include=object).columns
        num_df=df.select_dtypes(exclude=object).columns
        num_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df[cat_df]=cat_imputer.fit_transform(df[cat_df])
        df[num_df]=num_imputer.fit_transform(df[num_df])



    x=df.drop(target,axis=1)
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

    trans=make_column_transformer((label,df.select_dtypes(include=object).columns),remainder='passthrough')
    trans.fit_transform(df)
    pipe=Pipeline([('transform',trans),('scale',scale),('clf',model)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    target_names=df[target].unique()
    #fig=sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    return classification_report(y_test,y_pred,target_names=target_names, output_dict=True)


option = st.sidebar.selectbox(
    'Enough',
    ('Machine Learning','Deep Learning')
)
#st.sidebar.title("Enough")
pages = [
    "Introduction","Preprocessing","Sampling","Models","Performance Metrics"
    ,"Final code","Interview Q&A",'PlayGround','Support My Blog',
]

if option == 'Machine Learning':
    page = st.sidebar.radio('go to',pages)

    if page == "Introduction":
        st.markdown('<h1 style="text-decoration: underline;">ML Tree</h1>', unsafe_allow_html=True)
        mltree = Image.open("Image/mltree.png")
        st.image(mltree, use_column_width=True)
        st.markdown('<h1 style="text-decoration: underline;">Underfit & Overfit</h1>', unsafe_allow_html=True)
        uando = Image.open("Image/u&o.png")
        st.image(uando, use_column_width=True)
        st.markdown('<h1 style="text-decoration: underline;">Bias & Variance</h1>', unsafe_allow_html=True)
        bias = Image.open("Image/bias.png")
        st.image(bias, use_column_width=True)
    elif page=="Performance Metrics":
        st.markdown('<h1 style="text-decoration: underline;">Regression</h1>', unsafe_allow_html=True)
        
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

        st.markdown('<h1 style="text-decoration: underline;">Classification</h1>', unsafe_allow_html=True)

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
        st.markdown('<h1 style="text-decoration: underline;">Encoding</h1>', unsafe_allow_html=True)
        
        st.write("<h5>Label Encoder<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.preprocessing import LabelEncoder""", language='python')
        
        st.write("<h5>One-Hot Encoding<h5>",unsafe_allow_html=True)
        st.code("""import pandas as pd\npd.get_dummies(df['column'])""", language='python')
        
        st.markdown('<h1 style="text-decoration: underline;">Scaling</h1>', unsafe_allow_html=True)
        
        st.write("<h5>Standard Scaler<h5>",unsafe_allow_html=True)
        st.latex(r"y=\frac{x-mean}{Standard Deviation}")
        st.code("""from sklearn.preprocessing import StandardScaler""", language='python')

        st.write("<h5>Min-Max Scaler<h5>",unsafe_allow_html=True)
        st.latex(r"y=\frac{x-min}{max-min}")
        st.code("""from sklearn.preprocessing import MinMaxScaler""", language='python')

    elif page=="Sampling":
        st.markdown('<h1 style="text-decoration: underline;">Hold out</h1>', unsafe_allow_html=True)
        hold = Image.open("Image/hold_one.jpeg")
        st.image(hold, use_column_width=True)
        st.code("""from sklearn.model_selection import train_test_split""", language='python')

        st.markdown('<h1 style="text-decoration: underline;">Leave One Out</h1>', unsafe_allow_html=True)
        leave = Image.open("Image/leave.png")
        st.image(leave, use_column_width=True)
        st.code("""from sklearn.model_selection import LeaveOneOut""", language='python')

        st.markdown('<h1 style="text-decoration: underline;">K-Fold</h1>', unsafe_allow_html=True)
        kfold = Image.open("Image/k-fold.png")
        st.image(kfold, use_column_width=True)
        st.code("""from sklearn.model_selection import KFold""", language='python')

        st.markdown('<h1 style="text-decoration: underline;">Stratified K-Fold</h1>', unsafe_allow_html=True)
        skfold= Image.open("Image/stratified.png")
        st.image(skfold, use_column_width=True)
        st.code("""from sklearn.model_selection import StratifiedKFold""", language='python')

    elif page=="Models":
        st.markdown('<h1 style="text-decoration: underline;">Regression</h1>', unsafe_allow_html=True)
        
        st.write("<h5>Linear Regression<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.linear_model import LinearRegression""", language='python')
        
        st.write("<h5>Decision Tree<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.tree import DecisionTreeRegressor""", language='python')
        
        st.write("<h5>LassoCV<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.linear_model import LassoCV""", language='python')

        st.write("<h5>Support Vector Machines<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.svm import SVR""", language='python')

        st.markdown('<h1 style="text-decoration: underline;">Classification</h1>', unsafe_allow_html=True)

        st.write("<h5>Logistic Regression<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.linear_model import LogisticRegression""", language='python')
        
        st.write("<h5>Naïve Bayes<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB""", language='python')
        
        st.write("<h5>Decision Tree<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.tree import DecisionTreeClassifier""", language='python')

        st.write("<h5>Support Vector Machines<h5>",unsafe_allow_html=True)
        st.code("""from sklearn.svm import SVC""", language='python')
    elif page=="Final code":
        st.markdown('<h1 style="text-decoration: underline;">Regression</h1>', unsafe_allow_html=True)
        regression="""#Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score ,mean_squared_error ,mean_absolute_error
from sklearn import datasets

#Loading Data
x = pd.DataFrame(datasets.load_diabetes().data,columns=datasets.load_diabetes().feature_names)
y=pd.DataFrame(datasets.load_diabetes().target,columns=['diabetes'])

#Preprocessing
scale=MinMaxScaler()
label=OrdinalEncoder()
x[x.select_dtypes(exclude='object').columns]=scale.fit_transform(x.select_dtypes(exclude='object'))
x[x.select_dtypes(include='object').columns]=label.fit_transform(x.select_dtypes(include='object'))

#Sampling
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

#Modeling
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Performance Metrics
print(f'''Accuracy: {r2_score(y_test,y_pred)}
MSE: {mean_squared_error(y_test,y_pred)}
MAE: {mean_absolute_error(y_test,y_pred)}''')
    """
        st.code(regression, language='python')

        st.markdown('<h1 style="text-decoration: underline;">Classification</h1>', unsafe_allow_html=True)
        classification="""#Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import classification_report

#Loading Data
x = pd.DataFrame(datasets.load_iris().data,columns=datasets.load_iris().feature_names)
y=pd.DataFrame(datasets.load_iris().target,columns=['target'])

#Preprocessing
scale=MinMaxScaler()
label=OrdinalEncoder()
x[x.select_dtypes(exclude='object').columns]=scale.fit_transform(x.select_dtypes(exclude='object'))
x[x.select_dtypes(include='object').columns]=label.fit_transform(x.select_dtypes(include='object'))

#Sampling
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

#Modeling
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Performance Metrics
print(classification_report(y_test,y_pred))
    """
        st.code(classification, language='python')
    elif page=='Support My Blog':
        st.write("If you enjoy the content and find it valuable, consider supporting my blog to keep it running. Your contributions help to host playground, allow for more frequent updates, and support the creation of new content.")
        st.write(
        """
        Ways to Support:

        > Send me your feedback on [Gmail](mailto:abhishekjanjal1011@gmail.com])\n
        > connect me on [Linkedin](https://www.linkedin.com/in/abhishek-janjal-4329a7218/)\n
        > Donate via Gpay
        """)
        scan = Image.open("Image/scan.jpeg")
        st.image(scan, width=400)
        
    elif page=='PlayGround':
        option1=st.selectbox('Machine Learning',('Regression','Classification'))
        if option1=='Regression':
            uploaded_file = st.file_uploader("Choose a Dataset")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(df.head())
                if df is not None:
                    treat = st.selectbox("Cleaning Techinque", ["Drop", "Fill"])
                    
                # st.write("Feature Columns")
                # if df is not None:
                #     option2 = st.multiselect("Select options", df.columns, key="feature_columns")
                #     feature_dict={option: True for option in option2}
                #     x = df[feature_dict.keys()]
                if df is not None:
                    option3 = st.selectbox("Target Column", df.columns, key="target_column")
                    target=option3

                if df is not None:
                    col4, col5, col6 = st.columns([1, 1, 1])

                with col4:
                    option4 = st.selectbox("Label", ["Label Encode"])
                    if option4 == "Label Encode":
                        label = OrdinalEncoder()

                with col5:
                    option5 = st.selectbox("Scaling", ["MinMaxScaler", "StandardScaler"])
                    if option5 == "MinMaxScaler":
                        scale = MinMaxScaler()
                    elif option5 == "StandardScaler":
                        scale = StandardScaler()

                with col6:
                    option6 = st.selectbox("Model", ["Linear Regression", "Decision Tree", "LassoCV", "Support Vector Machines"])
                    if option6 == "Linear Regression":
                        model = LinearRegression()
                    elif option6 == "Decision Tree":
                        model = DecisionTreeRegressor()
                    elif option6 == "LassoCV":
                        model = LassoCV()
                    elif option6 == "Support Vector Machines":
                        model = SVR()

                r2,mse,mae=Train_reg(df,treat,'mean',target,label,scale,model)

                st.markdown(
                f"""
                <div style='color: green; font-size: 20px;'>
                    Accuracy:  {r2*100:.4f}<br>
                    Mean Squared Error:  {mse:.4f}<br>
                    Mean Absolute Error:  {mae:.4f}<br>
                    Root Mean Squared Error: {np.sqrt(mse):.4f}
                </div>
                """, 
                unsafe_allow_html=True
            )
                final_code=f"""## Importing Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,SVC
from  sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

## Loading Data

#df=pd.read_csv(file_name)

## Drop
# for i in df.columns[df.isnull().any()]:
#     df.drop(df[df[i].isnull()].index,inplace=True)

## Fill
# cat_df=df.select_dtypes(include=object).columns
# num_df=df.select_dtypes(exclude=object).columns
# num_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
# cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# df[cat_df]=cat_imputer.fit_transform(df[cat_df])
# df[num_df]=num_imputer.fit_transform(df[num_df])

x=df.drop('{target}',axis=1)
y=df['{target}']

## Preprocessing
scale={scale}
label=OrdinalEncoder()
x[x.select_dtypes(exclude='object').columns]=scale.fit_transform(x.select_dtypes(exclude='object'))
x[x.select_dtypes(include='object').columns]=label.fit_transform(x.select_dtypes(include='object'))

## Sampling
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

## Modeling
model={model}
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

## Performance Metrics
print(f'''Accuracy: {{r2_score(y_test,y_pred)}}
Mean Squared Error:  {{mean_squared_error(y_test,y_pred)}}
Mean Absolute Error:  {{mean_absolute_error(y_test,y_pred)}}
Root Mean Squared Error: {{np.sqrt(mean_squared_error(y_test,y_pred)):.4f}}''')
            """
                st.download_button('Download Code File', final_code, file_name='code.py')

        else:
                uploaded_file = st.file_uploader("Choose a Dataset")
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write(df.head())
                    if df is not None:
                        treat = st.selectbox("Cleaning Techinque", ["Drop", "Fill"])
                        
                    # st.write("Feature Columns")
                    # if df is not None:
                    #     option2 = st.multiselect("Select options", df.columns, key="feature_columns")
                    #     feature_dict={option: True for option in option2}
                    #     x = df[feature_dict.keys()]
                    if df is not None:
                        option3 = st.selectbox("Target Column", df.columns, key="target_column")
                        target=option3

                    if df is not None:
                        col4, col5, col6 = st.columns([1, 1, 1])

                    with col4:
                        option4 = st.selectbox("Label", ["Label Encode"])
                        if option4 == "Label Encode":
                            label = OrdinalEncoder()

                    with col5:
                        option5 = st.selectbox("Scaling", ["MinMaxScaler", "StandardScaler"])
                        if option5 == "MinMaxScaler":
                            scale = MinMaxScaler()
                        elif option5 == "StandardScaler":
                            scale = StandardScaler()

                    with col6:
                        option6 = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "GaussianNB", "Support Vector Machines","xgboost"])
                        if option6 == "Logistic Regression":
                            model = LogisticRegression()
                        elif option6 == "Decision Tree":
                            model = DecisionTreeRegressor()
                        elif option6 == "GaussianNB":
                            model = GaussianNB()
                        elif option6=="xgboost":
                            model=XGBClassifier()
                        elif option6 == "Support Vector Machines":
                            model = SVC()

                    result=Train_class(df,treat,'mean',target,label,scale,model)
                    
                    st.dataframe(result)
                    final_code=f"""## Importing Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from  sklearn.metrics import classification_report

## Loading Data

#df=pd.read_csv(file_name)

## Drop
# for i in df.columns[df.isnull().any()]:
#     df.drop(df[df[i].isnull()].index,inplace=True)

## Fill
# cat_df=df.select_dtypes(include=object).columns
# num_df=df.select_dtypes(exclude=object).columns
# num_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
# cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# df[cat_df]=cat_imputer.fit_transform(df[cat_df])
# df[num_df]=num_imputer.fit_transform(df[num_df])

x=df.drop('{target}',axis=1)
y=df['{target}']

##Preprocessing
scale={scale}
label=OrdinalEncoder()
x[x.select_dtypes(exclude='object').columns]=scale.fit_transform(x.select_dtypes(exclude='object'))
x[x.select_dtypes(include='object').columns]=label.fit_transform(x.select_dtypes(include='object'))

##Sampling
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

##Modeling
model={model}
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Performance Metrics
print(classification_report(y_test,y_pred))
            """
                    st.download_button('Download Code File', final_code, file_name='code.py')
    elif page=="Interview Q&A":
        st.write("<h5>1) What's the trade-off between bias and variance?</h5>",unsafe_allow_html=True)
        st.write(">If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.")
        
        st.write("<h5>2) What is gradient descent? </h5>",unsafe_allow_html=True)
        st.write(">Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.")

        st.write("<h5>3) Explain over- and under-fitting and how to combat them?</h5>",unsafe_allow_html=True)
        st.write(""">ML/DL models essentially learn a relationship between its given inputs(called training features) and objective outputs(called labels). Regardless of the quality of the learned relation(function), its performance on a test set(a collection of data different from the training input) is subject to investigation.

Most ML/DL models have trainable parameters which will be learned to build that input-output relationship. Based on the number of parameters each model has, they can be sorted into more flexible(more parameters) to less flexible(less parameters).

The problem of Underfitting arises when the flexibility of a model(its number of parameters) is not adequate to capture the underlying pattern in a training dataset. Overfitting, on the other hand, arises when the model is too flexible to the underlying pattern. In the later case it is said that the model has “memorized” the training data.

An example of underfitting is estimating a second order polynomial(quadratic function) with a first order polynomial(a simple line). Similarly, estimating a line with a 10th order polynomial would be an example of overfitting.""")

        st.write("<h5>4) How do you combat the curse of dimensionality? </h5>",unsafe_allow_html=True)
        st.write(""">Feature Selection(manual or via statistical methods)\n
>Principal Component Analysis (PCA)\n
>Multidimensional Scaling\n
>Locally linear embedding""")

        st.write("<h5>5) What is regularization, why do we use it, and give some examples of common methods?</h5>",unsafe_allow_html=True)
        st.write(""">A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. Examples

>Ridge (L2 norm)\n
>Lasso (L1 norm)
The obvious disadvantage of ridge regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the final model will include all predictors. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.""")

        st.write("<h5>6) Explain Principal Component Analysis (PCA)? </h5>",unsafe_allow_html=True)
        st.write(""">Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning to reduce the number of features in a dataset while retaining as much information as possible. It works by identifying the directions (principal components) in which the data varies the most, and projecting the data onto a lower-dimensional subspace along these directions.""")

        st.write("<h5>7) Why do we need a validation set and test set? What is the difference between them? </h5>",unsafe_allow_html=True)
        st.write(""">When training a model, we divide the available data into three separate sets:

>The training dataset is used for fitting the model’s parameters. However, the accuracy that we achieve on the training set is not reliable for predicting if the model will be accurate on new samples.\n
>The validation dataset is used to measure how well the model does on examples that weren’t part of the training dataset. The metrics computed on the validation data can be used to tune the hyperparameters of the model. However, every time we evaluate the validation data and we make decisions based on those scores, we are leaking information from the validation data into our model. The more evaluations, the more information is leaked. So we can end up overfitting to the validation data, and once again the validation score won’t be reliable for predicting the behaviour of the model in the real world.\n
>The test dataset is used to measure how well the model does on previously unseen examples. It should only be used once we have tuned the parameters using the validation set.\n
>So if we omit the test set and only use a validation set, the validation score won’t be a good estimate of the generalization of the model.""")

        st.write("<h5>8) What is stratified cross-validation and when should we use it?</h5>",unsafe_allow_html=True)
        st.write(""">Cross-validation is a technique for dividing data between training and validation sets. On typical cross-validation this split is done randomly. But in stratified cross-validation, the split preserves the ratio of the categories on both the training and validation datasets.

For example, if we have a dataset with 10% of category A and 90% of category B, and we use stratified cross-validation, we will have the same proportions in training and validation. In contrast, if we use simple cross-validation, in the worst case we may find that there are no samples of category A in the validation set.
Stratified cross-validation may be applied in the following scenarios:

On a dataset with multiple categories. The smaller the dataset and the more imbalanced the categories, the more important it will be to use stratified cross-validation.
On a dataset with data of different distributions. For example, in a dataset for autonomous driving, we may have images taken during the day and at night. If we do not ensure that both types are present in training and validation, we will have generalization problems.""")

        st.write("<h5>9) Can you explain the differences between supervised, unsupervised, and reinforcement learning?</h5>",unsafe_allow_html=True)
        st.write(""">In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.

With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model.

In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.""")

        st.write("<h5>10) What's the difference between boosting and bagging?</h5>",unsafe_allow_html=True)
        st.write(""">Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training.""")

        st.write("<h5></h5>",unsafe_allow_html=True)
        st.write(""">""")

        st.write("<h5></h5>",unsafe_allow_html=True)
        st.write(""">""")
    else:
        st.header(page)
        st.write("Content for " + page + " will be added here.")
elif option == 'Deep Learning':
    st.write("<h2>Coming Soon....</h2>",unsafe_allow_html=True)

 
