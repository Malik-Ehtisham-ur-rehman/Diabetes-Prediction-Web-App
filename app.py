# Required installations
# pip install streamlit
# pip install pandas
# pip install sklearn
# pip install matplotlib
# pip install seaborn
# pip install plotly

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# PAGE CONFIGURATION
st.set_page_config(page_title='Diabetes Prediction App', page_icon=':hospital:', layout='wide')

# LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except FileNotFoundError:
        st.error("Error: diabetes.csv file not found. Please ensure the file is in the correct directory.")
        return None

# PREPARE DATA
@st.cache_data
def prepare_data(df):
    # Separate features and target
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# USER INPUT FUNCTION
def user_report():
    st.sidebar.header('Patient Data Input')
    
    # Sliders for each feature with appropriate ranges
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 3)
    glucose = st.sidebar.slider('Glucose Level', 0, 250, 125)
    bp = st.sidebar.slider('Blood Pressure', 0, 150, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin Level', 0, 900, 79)
    bmi = st.sidebar.slider('Body Mass Index (BMI)', 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 15, 90, 35)

    # Create dictionary with correct column names
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Convert to DataFrame
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# MAIN APP FUNCTION
def main():
    # Title
    st.title('Diabetes Prediction Web App')
    
    # Load data
    df = load_data()
    
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Display dataset statistics
    st.subheader('Training Dataset Statistics')
    st.write(df.describe())
    
    # User input
    user_data = user_report()
    st.subheader('Your Input Data')
    st.write(user_data)
    
    # Train Model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Prediction
    user_result = rf.predict(user_data)
    prediction_proba = rf.predict_proba(user_data)
    
    # Display Prediction
    st.subheader('Prediction Result')
    if user_result[0] == 0:
        st.success('You are not likely to have diabetes.')
        color = 'blue'
    else:
        st.warning('You might be at risk of diabetes.')
        color = 'red'
    
    # Probability display
    st.write(f"Probability of being diabetic: {prediction_proba[0][1]*100:.2f}%")
    
    # Model Accuracy
    model_accuracy = accuracy_score(y_test, rf.predict(X_test)) * 100
    st.write(f"Model Accuracy: {model_accuracy:.2f}%")
    
    # VISUALIZATIONS
    st.title('Comparative Visualizations')
    
    # List of features to visualize
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create visualizations
    for feature in features:
        st.header(f'{feature} Comparison')
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Age', y=feature, hue='Outcome', data=df, palette='viridis')
        plt.scatter(user_data['Age'], user_data[feature], color=color, s=200, label='Your Data')
        plt.title(f'Age vs {feature} (Blue: Healthy, Red: Diabetic Risk)')
        plt.legend()
        st.pyplot(fig)
        plt.close(fig)

# Run the app
if __name__ == '__main__':
    main()