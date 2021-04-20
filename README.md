# Penguin Classification

0. [Creamos el archivo `penguins-model-building.py` ](#schema0)
<hr>

###  Creamos la app web, `penguins-app.py`
<hr>



<hr>

<a name="schema0"></a>

# 0. Creamos el archivo `penguins-model-building.py`

En este archivo creamos el modelo de predicción de los pingüinos.
Guardamos el modelo en un archivo `pickle`  para poder usar el modelo y no tener que estar ejecutando siempre el arhcivo `penguins-model-building.py`

<hr>

###  Creamos la app web, `penguins-app.py`
1. [Importamos librerías](#schema1)
2. [Títulos](#schema2)
3. [Recopila las características de entrada del usuario en el marco de datos](#schema3)
4. [Combinar las funciones de entrada del usuario con un conjunto de datos completo de pingüinos](#schema4)
5. [Codificación de característcas ordinales](#schema5)
6. [Mostramos lo valores del usuario](#schema6)
7. [Leemos el archivo salvado con el modelo y aplicamos el modelor y obtenemos predicciones](#schema7)
8. [Mostramos las predicciones](#schema8)

<hr>

<a name="schema1"></a>

# 1 . Importamos librerías
~~~python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
~~~

<hr>

<a name="schema2"></a>

# 2. Títulos
Título principal
~~~python
st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")
~~~
Títulos de la barra lateral
~~~Python
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")
~~~

<hr>

<a name="schema3"></a>

# 3. Recopila las características de entrada del usuario en el marco de datos

Primero creamos la función que va a coger los valores introducidos por el usuario en la barra lateral.

~~~python
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
~~~

Segundo comprobamos que el usuario ha usado el método de introducir un archivo
~~~python
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
~~~
Si no es así, usamos la función anteriomente declarada
~~~Python
else:
   
    input_df = user_input_features()
~~~

<hr>

<a name="schema4"></a>

# 4. Combinar las funciones de entrada del usuario con un conjunto de datos completo de pingüinos

~~~python
penguins_raw = pd.read_csv('./data/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)
~~~
<hr>

<a name="schema5"></a>

# 5. Codificación de característcas ordinales
~~~python
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Seleccionamos solo la primera fila(entrada de datos del usuario)
~~~

<hr>

<a name="schema6"></a>

# 6. Mostramos lo valores del usuario
~~~python
st.subheader('User Input features')
~~~

<hr>

<a name="schema7"></a>

# 7. Leemos el archivo salvado con el modelo y aplicamos el modelor y obtenemos predicciones
~~~python
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
~~~

<hr>

<a name="schema8"></a>

# 8. Mostramos las predicciones
~~~python
st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
~~~








# Documentación
https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

https://www.youtube.com/watch?v=Eai1jaZrRDs

https://github.com/dataprofessor/code/tree/master/streamlit/part3