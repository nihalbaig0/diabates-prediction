# This is a sample Python script.
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)
def preprocess(df):
    columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    np.random.seed(88)

    for column in columns:
        mean = df[column].mean()
        std = df[column].std()

        col_values = df[column].values

        for i, val in enumerate(col_values):
            if val == 0:
                col_values[i] = mean + (2 * std * np.random.random() - 0.5)

        df[column] = pd.Series(col_values)
    return df

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    st.write("""
    # Diabetes Prediction App (AI Project #Team 8)
    This app predicts  **Diabetes**! You can give your custom input in sidebar.
    We used *Random Forest Model* to predict **Diabetes**
    """)
    st.write('---')

    # Loads the Boston House Price Dataset

    df = pd.read_csv('./diabetes.csv')
    df = preprocess(df)
    X = df.drop('Outcome', axis=1)
    Y = df['Outcome']
    st.sidebar.header('Specify Input Parameters')
    st.sidebar.subheader("Give your own inputs to try this model")
    def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies', float(X.Pregnancies.min()), float(X.Pregnancies.max()), float(X.Pregnancies.mean()))
        Glucose = st.sidebar.slider('Glucose', float(X.Glucose.min()), float(X.Glucose.max()), float(X.Glucose.mean()))
        BloodPressure = st.sidebar.slider('BloodPressure', float(X.BloodPressure.min()), float(X.BloodPressure.max()), float(X.BloodPressure.mean()))
        SkinThickness = st.sidebar.slider('SkinThickness', float(X.SkinThickness.min()), float(X.SkinThickness.max()), float(X.SkinThickness.mean()))
        Insulin = st.sidebar.slider('NOX', float(X.Insulin.min()), float(X.Insulin.max()), float(X.Insulin.mean()))
        BMI = st.sidebar.slider('BMI', float(X.BMI.min()), float(X.BMI.max()), float(X.BMI.mean()))
        DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', float(X.DiabetesPedigreeFunction.min()), float(X.DiabetesPedigreeFunction.max()), float(X.DiabetesPedigreeFunction.mean()))
        Age = st.sidebar.slider('Age', float(X.Age.min()), float(X.Age.max()), float(X.Age.mean()))
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    # Main Panel

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')

    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, Y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)

    st.header('Prediction of Outcome')
    st.write(prediction)
    st.write('---')

    # Explaining the model's predictions using SHAP values
    # https://github.com/slundberg/shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')

    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    print(df.describe())  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
