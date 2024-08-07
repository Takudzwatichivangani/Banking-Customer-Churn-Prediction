#importing required packages
import streamlit as st
import pandas as pd
from keras.models import load_model

# Loading model
model = load_model('Ann.keras')

# Defining label encoder mappings
gender_mapping = {'Male': 0, 'Female': 1}
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}

st.header('Bank Customer Churn Prediction')

# Input fields
st.sidebar.header('Enter Customer Information')
credit_score = st.sidebar.number_input('Credit Score', min_value=0)
age = st.sidebar.slider('Age', min_value=18, max_value=100)
tenure = st.sidebar.number_input('Tenure', min_value=0)
balance = st.sidebar.number_input('Balance', min_value=0)
num_of_products = st.sidebar.slider('Number Of Products', min_value=1, max_value=4)
has_credit_card = st.sidebar.selectbox('Has Credit Card', ('Yes', 'No'))
is_active_member = st.sidebar.selectbox('Is Active Member', ('Yes', 'No'))
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0)

# Dropdowns
st.sidebar.header('Additional Information')
geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))

# Performing churn prediction based on inputs
if st.sidebar.button('Predict'):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumberOfProducts': [num_of_products],
        'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
        'Gender': [gender]
    })

    # Preprocessing the input data
    input_data['Gender'] = input_data['Gender'].map(gender_mapping).astype(int)
    input_data['Geography'] = input_data['Geography'].map(geography_mapping).astype(int)

    # Make the prediction
    prediction = model.predict(input_data)

    # Displaying the prediction result
    result = 'churn' if prediction[0] == 1 else 'remain with the bank'
    st.subheader('Prediction Result')
    st.write('The customer is likely to', result)



st.write('''
 ### About Customer Churn in Banking
In the banking sector, customer churn is a critical issue that can significantly impact a bank's profitability and growth. Churn occurs when customers close their accounts and move to competitors, often due to dissatisfaction with services or better offers elsewhere. Identifying potential churners in advance allows banks to take proactive measures to retain these customers, thereby reducing churn rates and maintaining a stable customer base.
 
This application leverages a deep learning model, specifically an Artificial Neural Network (ANN), to predict whether a bank customer is likely to churn based on their profile information. ANNs are powerful machine learning models inspired by the human brain's neural networks, capable of capturing complex patterns and relationships in data. By inputting various customer attributes such as credit score, age, tenure, balance, number of products, credit card status, active member status, estimated salary, geography, and gender, the ANN model processes these features and outputs a prediction indicating the likelihood of customer churn. This predictive capability enables banks to implement targeted retention strategies and improve customer satisfaction.
''')

# Displaying the entered information
info = {
    'Credit Score': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'Number of Products': num_of_products,
    'Has Credit Card': has_credit_card,
    'Is Active Member': is_active_member,
    'Estimated Salary': estimated_salary,
    'Geography': geography,
    'Gender': gender
}

df_info = pd.DataFrame(info, index=[0])


df_info['Credit Score'] = df_info['Credit Score'].astype(int)
df_info['Age'] = df_info['Age'].astype(int)
df_info['Tenure'] = df_info['Tenure'].astype(int)
df_info['Balance'] = df_info['Balance'].astype(float)
df_info['Number of Products'] = df_info['Number of Products'].astype(int)
df_info['Has Credit Card'] = df_info['Has Credit Card'].map({'Yes': 1, 'No': 0}).astype(int)
df_info['Is Active Member'] = df_info['Is Active Member'].map({'Yes': 1, 'No': 0}).astype(int)
df_info['Estimated Salary'] = df_info['Estimated Salary'].astype(float)
df_info['Geography'] = df_info['Geography'].map(geography_mapping).astype(int)
df_info['Gender'] = df_info['Gender'].map(gender_mapping).astype(int)

st.write('### Entered Information')
st.table(df_info.T)  

