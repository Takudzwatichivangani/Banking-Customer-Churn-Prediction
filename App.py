import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib  # Import joblib to load the scaler

# Loading the trained ANN model
model = tf.keras.models.load_model('Ann.keras')

# Loading the StandardScaler used during training
scaler = joblib.load('scaler.pkl')

# Defining label encoder mappings
gender_mapping = {'Male': 0, 'Female': 1}
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}

st.header('Bank Customer Churn Prediction')

# Displaying an image
st.image('https://uxpressia.com/blog/wp-content/uploads/2022/12/Frame-20-640x339.png')

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

# Dropdowns for additional information
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
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
        'Gender': [gender]
    })

    # Preprocessing the input data
    input_data['Gender'] = input_data['Gender'].map(gender_mapping).astype(int)
    input_data['Geography'] = input_data['Geography'].map(geography_mapping).astype(int)

    # Standardize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)
    predicted_prob = prediction[0][0]  # Assuming the output is a probability
    result = 'churn' if predicted_prob > 0.5 else 'remain with the bank'

    # Display the prediction result
    st.subheader('Prediction Result')
    st.write(f'The customer is likely to {result}, with a probability of churning at {predicted_prob:.6f}.')

# File upload for bulk prediction
st.sidebar.header('Bulk Prediction')
uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is not None:
    try:
        # Reading the uploaded CSV file
        bulk_data = pd.read_csv(uploaded_file)

        # Checking the columns in the uploaded file
        st.write('Uploaded CSV File Preview:')
        st.write(bulk_data.head())

        # Selecting only the columns required for prediction
        required_columns = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender'
        ]

        # Handling missing columns by filling with default values
        for column in required_columns:
            if column not in bulk_data.columns:
                st.warning(f'Missing column: {column}. Adding default values.')
                bulk_data[column] = 0

        # Filtering only required columns
        bulk_data = bulk_data[required_columns]

        # Preprocessing the bulk data
        bulk_data['Gender'] = bulk_data['Gender'].map(gender_mapping).astype(int)
        bulk_data['Geography'] = bulk_data['Geography'].map(geography_mapping).astype(int)

        # Standardize the bulk data using the loaded scaler
        bulk_data_scaled = scaler.transform(bulk_data)

        # Making bulk predictions
        bulk_predictions = model.predict(bulk_data_scaled)

        # Adding prediction results to the bulk data
        bulk_data['Churn Probability'] = bulk_predictions
        bulk_data['Prediction'] = bulk_data['Churn Probability'].apply(lambda x: 'churn' if x > 0.5 else 'remain with the bank')

        st.subheader('Bulk Prediction Results')
        st.write(bulk_data)
        st.download_button(label='Download Predictions', data=bulk_data.to_csv(index=False), file_name='bulk_predictions.csv', mime='text/csv')
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write('''
### About Customer Churn in Banking
In the banking sector, customer churn is a critical issue that can significantly impact a bank's profitability and growth. Churn occurs when customers close their accounts and move to competitors, often due to dissatisfaction with services or better offers elsewhere. Identifying potential churners in advance allows banks to take proactive measures to retain these customers, thereby reducing churn rates and maintaining a stable customer base.
 
This application leverages a deep learning model, specifically an Artificial Neural Network (ANN), to predict whether a bank customer is likely to churn based on their profile information. ANNs are powerful machine learning models inspired by the human brain's neural networks, capable of capturing complex patterns and relationships in data. By inputting various customer attributes such as credit score, age, tenure, balance, number of products, credit card status, active member status, estimated salary, geography, and gender, the ANN model processes these features and outputs a prediction indicating the likelihood of customer churn. This predictive capability enables banks to implement targeted retention strategies and improve customer satisfaction.
''')

 

