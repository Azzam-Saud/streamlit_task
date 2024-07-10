import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Insurance Charges Prediction")
    st.write("""
    This app predicts insurance charges based on user input using a Linear Regression model.
    """)
    
    # Load the dataset
    @st.cache
    def load_data():
        df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\insurance.csv')
        return df
    
    df = load_data().copy()  # Make a copy of the loaded data

    # Display the dataset
    # st.subheader("Dataset")
    # st.write(df.head())

    # Encoding categorical features
    features = ['sex', 'smoker']
    LE = LabelEncoder()
    for f in features:
        df[f] = LE.fit_transform(df[f])
    
    # Split the data into features and target
    X = df.drop(['charges', 'region'], axis=1)
    y = df['charges']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    SS = StandardScaler()
    X_train = SS.fit_transform(X_train)
    X_test = SS.transform(X_test)

    # Train Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # User input for prediction
    st.subheader("Enter Feature Values for Prediction")
    
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    
    # Encode user inputs
    sex_encoded = LE.fit_transform([sex])[0]
    smoker_encoded = LE.fit_transform([smoker])[0]
    
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded]
    })
    
    # Standardize the user input
    user_data_scaled = SS.transform(user_data)

    # Make prediction
    if st.button("Predict"):
        predicted_charge = lin_reg.predict(user_data_scaled)[0]
        st.write(f"Predicted Insurance Charge: ${predicted_charge:.2f}")

if __name__ == "__main__":
    main()
