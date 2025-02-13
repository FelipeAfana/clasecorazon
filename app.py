import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Título de la aplicación
st.title("Asistente Cardiaco")

# Autor de la aplicación
st.markdown("**Autor**: Felipe Afandor")

# Párrafo de instrucciones
st.write("""
Esta aplicación permite ingresar datos sobre la **edad** y el **nivel de colesterol** de una persona. 
Basándose en estos datos, el modelo entrenado predice si la persona tiene o no problemas cardiacos. 
El modelo fue entrenado utilizando un clasificador **KNN** y fue normalizado con el **MinMaxScaler**.
Por favor, ingrese los datos en la pestaña de "Ingreso de Datos" y consulte el resultado en la pestaña "Resultado".
""")

# Cargar los modelos guardados
scaler = joblib.load("esclador.bin")
model = joblib.load("modelo_knn.bin")

# Crear dos pestañas
tab1, tab2 = st.tabs(["Ingreso de Datos", "Resultado"])

# Pestaña de ingreso de datos
with tab1:
    st.header("Ingreso de Datos")

    # Desplegar un slider para seleccionar la edad
    edad = st.slider('Selecciona la edad', min_value=18, max_value=80, value=30)

    # Desplegar un slider para seleccionar el colesterol
    colesterol = st.slider('Selecciona el nivel de colesterol', min_value=100, max_value=600, value=200)

    # Crear un DataFrame con los datos ingresados
    df = pd.DataFrame({'Edad': [edad], 'Colesterol': [colesterol]})

    # Mostrar el DataFrame
    st.write("DataFrame Ingresado")
    st.dataframe(df)

    # Almacenar los valores ingresados en sesión
    st.session_state.df_ingresado = df

# Pestaña de resultados
with tab2:
    st.header("Resultado de la Predicción")

    # Verificar si los datos fueron ingresados
    if 'df_ingresado' in st.session_state:
        # Obtener el dataframe de la sesión
        df_ingresado = st.session_state.df_ingresado

        # Normalizar los datos utilizando el MinMaxScaler
        df_normalizado = scaler.transform(df_ingresado)

        # Realizar la predicción utilizando el modelo KNN
        prediccion = model.predict(df_normalizado)

        # Mostrar el resultado
        if prediccion[0] == 1:
            st.write("La persona **tiene problemas cardiacos**.")
            st.image("https://static.wixstatic.com/media/737f5c_1dae298510b9435a8923b1be34509565~mv2.jpg/v1/fill/w_600,h_400,al_c,q_80,enc_avif,quality_auto/737f5c_1dae298510b9435a8923b1be34509565~mv2.jpg")
        else:
            st.write("La persona **no tiene problemas cardiacos**.")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5W37zVKhWj5DYNwf1k3h0eiX6Z269LdMF0w&s")
    else:
        st.write("Por favor, ingresa los datos en la primera pestaña.")
