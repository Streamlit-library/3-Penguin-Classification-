import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Recopila las características de entrada del usuario en el marco de datos
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    
    input_df = user_input_features()

# Combina las funciones de entrada del usuario con un conjunto de datos completo de pingüinos
# Esto será útil para la fase de codificación.
penguins_raw = pd.read_csv('./data/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Codificación de característcas ordinales
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Seleccionamos solo la primera fila(entrada de datos del usuario)

# Mostramos lo valores
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Leemos el archivo salvado con el modelo
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Aplicamos el modelor y obtenemos predicciones
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Mostramos las predicciones
st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)