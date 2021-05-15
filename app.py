import streamlit as st
import pickle
import numpy as np
from PIL import Image

logo = Image.open('logo.png')
image1 = Image.open('approvedd.png')
image2 = Image.open('napproved.png')

# App title
st.title("LOAN APPROVAL PREDICTION")
st.markdown('''
- This Web Application can help you to approve or reject a loan application.
''')
st.write('---')

# Sidebar
st.sidebar.subheader('INPUT PARAMETERS')

gender = st.sidebar.selectbox('Gender :', ('Male', 'Female'))
if gender == 'Male':
    gender_v = 1
else:
    gender_v = 0

married = st.sidebar.selectbox('Married Status :', ('Yes', 'No')) 
if married == 'Yes':
    married_v = 1
else:
    married_v = 0 

dependents = st.sidebar.slider('Dependents:', 0, 3) 

education = st.sidebar.selectbox('Education :', ('Graduate', 'Not Graduate')) 
if education == 'Graduate':
    education_v = 1
else:
    education_v = 0 

se = st.sidebar.selectbox('Self Employed :', ('Yes', 'No')) 
if se == 'Yes':
    se_v = 1
else:
    se_v = 0 
    
ai = st.sidebar.number_input('Applicant Income :', value=int(), step=int())
cai = st.sidebar.number_input('Coapplicant Income :', value=float(), step=float())
la = st.sidebar.number_input('Loan Amount :', value=float(), step=float())
lat = st.sidebar.selectbox('Loan Amount Term :', (36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0))
ch = st.sidebar.selectbox('Credit History :', (0.0, 1.0)) 

pa = st.sidebar.selectbox('Property Area :', ('Rural', 'Semi-Urban', 'Urban')) 
if pa == 'Rural':
    pa_v = 0
elif pa == 'Semi-Urban':
    pa_v = 1 
else:
    pa_v = 2 

Fitur = np.array([gender_v, married_v, dependents, education_v, se_v, ai, cai, la, lat, ch, pa_v])
filename = 'loan_predict.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(Fitur.reshape(1,-1))
if result == 0:
    r = 'Loan Not Approved'
    image = image2
else:
    r = 'Loan Approved'
    image = image1

st.image(logo)
if st.button('Process') :
    st.write('---')
    st.header('**Applicant Profile**')
    st.write('**Gender :**', gender)
    st.write('**Married Status :**', married)
    st.write('**Dependents :**', str(dependents))
    st.write('**Education :**', education)
    st.write('**Self Employed :**', se)
    st.write('**Applicant Income :**', str(ai))
    st.write('**Coapplicant Income :**', str(cai))
    st.write('**Loan Amount :**', str(la))
    st.write('**Loan Amount Term :**', str(lat))
    st.write('**Credit History :**', str(ch))
    st.write('**Property Area :**', pa)
    st.write('---')
    st.header('**Fitur After Transformation**')
    st.write(Fitur.reshape(1,-1))
    st.write('---')
    st.header('**Prediction Result**')
    st.write(result)
    st.subheader(r)
    st.image(image)
    
    


