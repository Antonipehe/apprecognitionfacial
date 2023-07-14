import streamlit as st
import pickle
import numpy as np
import requests
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import io
import json
import time




# Cargar los modelos de predicción neuronal
age_model = keras.models.load_model('models/AgesModel.h5')
emotion_model = keras.models.load_model('models/EmotionsModel.h5')

# Función para preprocesar la imagen antes de la predicción
def preprocess_image(image):
    # Preprocesamiento de la imagen (ajuste según sea necesario)
    processed_image = image.resize((150, 150))
    processed_image = img_to_array(processed_image)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Función para realizar la predicción de edad en una imagen
def predict_age(image):
    processed_image = preprocess_image(image)
    prediction = age_model.predict(processed_image)
    # Ajusta la lógica de predicción según tu modelo de edad
    age_range = int(prediction[0][0])

    if age_range <= 5:
        age_range = "0-5"
    elif age_range <= 10:
        age_range = "5-10"
    elif age_range <= 15:
        age_range = "10-15"
    elif age_range <= 20:
        age_range = "15-20"
    elif age_range <= 25:
        age_range = "20-25"
    elif age_range <= 30:
        age_range = "25-30"
    elif age_range <= 35:
        age_range = "30-35"
    elif age_range <= 40:
        age_range = "35-40"
    elif age_range <= 45:
        age_range = "40-45"
    elif age_range <= 50:
        age_range = "45-50"
    elif age_range <= 55:
        age_range = "50-55"
    elif age_range <= 60:
        age_range = "55-60"
    elif age_range <= 65:
        age_range = "60-65"
    elif age_range <= 70:
        age_range = "65-70"
    elif age_range <= 75:
        age_range = "70-75"
    elif age_range <= 80:
        age_range = "75-80"
    elif age_range <= 85:
        age_range = "80-85"
    elif age_range <= 90:
        age_range = "85-90"
    elif age_range <= 95:
        age_range = "90-95"
    elif age_range <= 100:
        age_range = "95-100"

    return age_range



# Función para realizar la predicción de emociones en una imagen
def predict_emotion(image):
    processed_image = preprocess_image(image)
    prediction = emotion_model.predict(processed_image)
    # Ajusta la lógica de predicción según tu modelo de emociones
    emotion = np.argmax(prediction[0])
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    prediction_emotion = emotions[emotion]

    return prediction_emotion

# Configuración de la página
st.set_page_config(
    page_title="APP FACE RECOGNITION",
    page_icon=":eyeglasses:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centrar el título de la aplicación
st.markdown("<h1 style='text-align: center;'>👓 APP FACE RECOGNITION</h1>", unsafe_allow_html=True)

# # Cargar el archivo JSON localmente
# with open('icono.json') as json_file:
#     lottie_json = json.load(json_file)
# st_lottie(lottie_json, height=200)


# Menú de navegación
pestañas = option_menu(None, ["Proyecto", "Modelo", "Planning", "Graficos"], 
    icons=['house', 'camera fill', 'kanban', 'book'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
pestañas

if pestañas == "Proyecto":
    # Contenido de la página "Proyecto"
    st.header("📝 Proyecto")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(
                """
                ### 🚩 Objetivos principales:
    
                - Entrenar una red neuronal utilizando una base de datos a partir de imágenes.
                - Reconocer y clasificar la expresión facial de las personas en las imágenes.
                - Determinar el rango de edad de las personas a partir de las imágenes.
                
                """
            )
        
        with right_column:
            # Leer el contenido del archivo JSON
            with open("imagenes/faceicon.json", "r") as json_file:
                json_data = json.load(json_file)
            # Mostrar la animación con el contenido del archivo JSON
            st_lottie(json_data, height=200, key="coding")
        
        descripcion = """
        #### 📋 Importancia y Aplicaciones
        
        Este proyecto tiene relevancia en varias áreas, como:
        
        - **Reconocimiento Facial**: Permite identificar y clasificar emociones en imágenes, lo cual resulta útil en aplicaciones de seguridad, análisis y sistemas de interacción humano-computadora.
        - **Publicidad Personalizada**: La clasificación del rango de edad obtenida a través del reconocimiento facial puede ayudar a adaptar los anuncios y contenidos en línea a diferentes grupos demográficos. Esto permite ofrecer una experiencia de usuario más personalizada y relevante.
        - **Investigación de Mercado**: El reconocimiento facial proporciona información valiosa sobre las preferencias y comportamientos de diferentes grupos de edad en relación con productos y servicios. Esto ayuda a las empresas a comprender mejor a su audiencia y tomar decisiones de marketing más informadas.
        
        
        #### 🔨 Proceso de Desarrollo
        
        El proyecto consta de los siguientes pasos:
        
        1. **Recopilación y Preparación de Datos**: Se utiliza una base de datos de imágenes etiquetadas con expresiones faciales y edades para entrenar el modelo. La calidad y cantidad de los datos son fundamentales para lograr resultados precisos y confiables.
        2. **Entrenamiento del Modelo**: Se entrena una red neuronal utilizando los datos recopilados y se ajusta su rendimiento. Esto implica alimentar el modelo con las imágenes etiquetadas y optimizar sus parámetros para que aprenda a reconocer patrones y características relevantes.
        3. **Validación y Evaluación**: Se evalúa el modelo utilizando datos de prueba que no se utilizaron durante el entrenamiento. Esto permite medir su precisión y desempeño, identificar posibles problemas y realizar ajustes necesarios.
        4. **Despliegue y Uso**: Se crea una interfaz de usuario interactiva en Streamlit para utilizar el modelo entrenado y realizar inferencias en nuevas imágenes. Esto facilita la aplicación práctica del reconocimiento facial y brinda una experiencia accesible y amigable para los usuarios.
        
        
        #### ✅ Ventajas del Reconocimiento Facial:
        
        - Mayor seguridad en el acceso a dispositivos y espacios físicos.
        - Personalización de contenidos y anuncios en línea, mejorando la experiencia del usuario.
        - Automatización de procesos en aplicaciones de análisis y sistemas de interacción.
        - Herramienta útil en investigaciones de mercado y análisis demográficos.
        
        
        #### ❌ Desventajas y Consideraciones Éticas:
        
        - Posibles riesgos de privacidad y protección de datos personales.
        - Potencial de discriminación y sesgos si no se manejan adecuadamente los datos y algoritmos.
        - Necesidad de regulaciones y políticas claras para proteger los derechos individuales.
        - Importancia de la transparencia y explicabilidad de los algoritmos utilizados.
        - Conciencia y mitigación de posibles impactos sociales y culturales.
        
        Es esencial abordar estas desventajas y consideraciones éticas para garantizar un uso responsable y ético del reconocimiento facial.
        """
        
        st.markdown(descripcion)


if pestañas == "Modelo":
    # Contenido de la página "Modelo"
    st.header("Modelo")
    st.write("Comenzar con la predicción 👇")

    # Formulario de carga de imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la animación de carga mientras se procesa la imagen
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Leer la imagen
        image = Image.open(uploaded_file)

        # Simular el tiempo de procesamiento
        for i in range(1, 101):
            time.sleep(0.1)
            progress_bar.progress(i)
            status_text.text(f"Procesando predicción de la imagen... {i}% completado")

        # Detener la animación y mostrar la imagen cargada
        progress_bar.empty()
        status_text.text("Imagen cargada")

        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)

        # Mostrar la imagen en la columna derecha
        with col2:
            st.image(image, caption='Imagen cargada', use_column_width=False)

        # Realizar la predicción de edad
        age_range = predict_age(image)

        # Realizar la predicción de emociones
        emotion = predict_emotion(image)

        # Mostrar los resultados en la columna izquierda
        with col1:
            st.subheader("Resultado de la predicción:")
            st.write(f"- **Expresión**: {emotion}")
            st.write(f"- **Edad**: {age_range}")

elif pestañas == "Planning":
    # Contenido de la página "Planning"
    with st.expander("Metodo SCRUM"):
    
    # Dividir la página en dos columnas
        left_column, right_column = st.columns([3, 1])

    # Columna izquierda: Descripción
        with left_column:
            descripcion = """
            ### Metodo SCRUM
            
            Diseñado para gestionar proyectos complejos, fomentar la colaboración y la adaptabilidad, y ofrecer un enfoque iterativo e incremental. 
    
            Puntos clave del método:
    
            - **Sprint:** Es el corazón del método Scrum. Es un período de tiempo fijo, generalmente de 1 a 4 semanas, durante el cual se lleva a cabo el trabajo. 
            - **Reunión de Planificación:** Al inicio, el equipo de desarrollo se reúne para seleccionar las tareas del y establecer los objetivos y la planificación.
            - **Reuniones Diarias:** Son breves reuniones diarias en las que el equipo de desarrollo comparte el progreso, identifica los obstáculos y coordina el trabajo restante.
            - **Revisión:** Al finalizar, cada miembro del equipo de desarrollo muestra el trabajo realizado, quienes brindan retroalimentación y hacen los ajustes necesarios.
            - **Análisis final:** Después de la Revisión, el equipo de desarrollo se reúne para analizar cómo funcionó y cómo pueden mejorar en el próximo. Se identifican acciones para implementar mejoras en el proceso.
            """
            st.markdown(descripcion)
            
        # Columna derecha: Imagen
        with right_column:
            # Cargar y mostrar la imagen
            image = Image.open("imagenes/scrum.png")
            st.image(image, use_column_width=True)

    # Desplegables
    with st.expander("Planning Semanal del Proyecto"):
        descripcion ="""
        #### 🗓️ Semana 1
    
        ##### Selección del proyecto: Reconocimiento facial
    
            Para llevar a cabo este proyecto, utilizaremos redes neuronales de tipo convolucional. 
            El proyecto consta de una base de datos llena de imágenes de diferentes personas con diferentes características. 
            El objetivo es lograr reconocer el rango de edad de la persona y sus expresiones.
    
        ##### Recolección y preparación de datos
    
        Tecnologías y librerías: Python, pandas o Polars, OpenCV, Cv2, etc.
        
            Pasos:
            - Recopilación de imágenes y asegurarnos de que estén en un formato compatible, como JPG o PNG.
            - Utilizar pandas para leer el archivo CSV y preprocesar los datos.
            - Utilizar OpenCV para leer y preprocesar las imágenes. Puedes necesitar realizar operaciones como el redimensionamiento, la normalización, la detección de caras, etc.
            - Almacenar las imágenes y sus características en una base de datos SQL. PostgreSQL es una opción popular para esto. Asegúrate de que las imágenes estén correctamente asociadas con sus características correspondientes.
    
        #### 🗓️ Semana 2
    
       ##### Modelado
        
       Tecnologías y librerías: TensorFlow, Keras, scikit-learn
        
            Pasos:
            - Utilizar TensorFlow y Keras para construir una red neuronal. Comenzar con una red neuronal convolucional (CNN), que es comúnmente usada para el reconocimiento visual.
            - Dividir los datos en conjuntos de entrenamiento y validación. Utilizar el conjunto de entrenamiento para entrenar tu red neuronal y el conjunto de prueba para evaluar su rendimiento.
            - Ajustar los parámetros de tu modelo para mejorar su rendimiento. Esto puede incluir la tasa de aprendizaje, el número de capas en la red, etc.
    
        ##### Evaluación del modelo
        
        Tecnologías y librerías: scikit-learn, matplotlib, seaborn
        
            Pasos:
            - Utilizar scikit-learn para calcular métricas de evaluación, como la precisión y la matriz de confusión.
            - Utilizar matplotlib y seaborn para visualizar los resultados de tu modelo. Esto puede incluir una curva ROC, una matriz de confusión, etc.
    
        #### 🗓️ Semana 3
    
        ##### Elaboración de la app con Streamlit:
            
       Tecnologías y librerías: TensorFlow y Streamlit.
    
            Una vez entrenados y guardados los modelos con los que vamos a trabajar, comenzamos con la elaboración de la app, 
            cuyo objetivo principal es que el usuario introduzca una imagen facial para predecir la edad y la expresión facial.
    
        #### 🗓️ Semana 4
    
        Prueba del modelo con imágenes al azar y mejoras de la app.
        
        """   
        st.markdown(descripcion)
        
    with st.expander("Notion"):
        # Dividir la página en dos columnas
        left_column, right_column = st.columns([3, 1])
    
        # Columna izquierda: Descripción
        with left_column:
            descripcion = """
            ### Notion
    
            **Notion** es una efectiva herramienta en la gestión de proyectos y el seguimiento de tiempos. 
            Mantener la información actualizada y utilizar las diferentes vistas y herramientas disponibles 
            permite tener una visión clara del progreso del proyecto y cumplir con los plazos establecidos.
    
            #### Principales características de Notion:
    
            - Integración de bibliotecas y herramientas de Python.
            - Automatización de tareas.
            - Incrustación de código Python en las páginas utilizando bloques de código, además de poder ejecutarlo.
            - Visualización de datos como tablas y gráficos utilizando las bibliotecas Matplotlib, Seaborn o Plotly para generar visualizaciones interactivas.
            - Colaboración con otros miembros del equipo en tiempo real.
    
            """
    
            st.markdown(descripcion)
    
        # Columna derecha: Imagen
        with right_column:
            # Cargar y mostrar la imagen
            image = Image.open("imagenes/notion.png")
            st.image(image, use_column_width=True)
            
import json

if pestañas == "Graficos":
    # Contenido de la página "Graficos"
    st.header("Graficos")
    st.write("#### Visualización del progreso del entrenamiento y validación")

    # Cargar las imágenes
    image1 = Image.open("imagenes/Imagen 1.png")
    image2 = Image.open("imagenes/Imagen 2.png")
    image3 = Image.open("imagenes/grafico3.png")
    image4 = Image.open("imagenes/grafico4.png")

    # Crear los botones
    boton1 = st.button("Reconocimiento edad")
    boton2 = st.button("Reconocimiento emociones")
    boton3 = st.button("Red Neuronal")

    # Mostrar el contenido según el botón seleccionado
    if boton1:
        # Mostrar las dos imágenes de la opción 1
        st.image(image1, use_column_width=False)
        st.image(image2, use_column_width=False)
    elif boton2:
        # Mostrar las dos imágenes de la opción 2
        st.image(image3, use_column_width=False)
        st.image(image4, use_column_width=False)
    elif boton3:
        # Ruta del archivo de video guardado localmente
        video_path = "imagenes/redneuronal.mp4"

        # Leer el contenido del archivo de video en bytes
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Establecer el estilo CSS para el reproductor de video
        video_style = """
        <style>
        video {
            max-width: 800px;
            width: 100%;
            height: auto;
        }
        </style>
        """

        # Mostrar el estilo CSS
        st.markdown(video_style, unsafe_allow_html=True)

        # Dividir la página en columnas
        left_column, right_column = st.columns([2, 1])

        # Mostrar el video en Streamlit en la columna izquierda
        with left_column:
            st.video(video_bytes, format='video/mp4')


        # Mostrar el contenido del archivo JSON en la columna derecha
        with right_column:
            # Leer el contenido del archivo JSON
            with open("imagenes/neuronas.json", "r") as json_file:
                json_data = json.load(json_file)
            # Mostrar la animación con el contenido del archivo JSON
            st_lottie(json_data, height=300, key="coding")


    
if __name__ == '__preprocess_image__':
    preprocess_image()
