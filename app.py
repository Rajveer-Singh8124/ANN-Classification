import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


#Load the trained model
model = tf.keras.models.load_model("data/model.h5")

# Load the encoder and scaler
with open("data/label_encoder_gender.pkl","rb") as f:
    label_encoder_gen = pickle.load(f)

with open("data/one_hot_encoder_gro.pkl","rb") as f:
    one_hot_encoder_geo = pickle.load(f)

with open("data/scaler.pkl","rb") as f:
    scaler = pickle.load(f)
    

st.title("Customer Churn Prediction")


geo = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gen.classes_)
age = st.slider("Age",18,92)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

input_df= pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encoder_gen.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
}
)

geo_encoded = one_hot_encoder_geo.transform([[geo]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out())


input_df = pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

input_df_scale = scaler.transform(input_df)

pred = model.predict(input_df_scale)
pred_proba = pred[0][0]

st.write(f"Churn Probability  {pred_proba:.2f} ")

if pred_proba > .5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")






