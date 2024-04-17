<h1 align="center"> PROYECTO INDIVIDUAL Nº1 </h1>

<img src="https://media.licdn.com/dms/image/C4D12AQGZ3XMPSxkgyg/article-cover_image-shrink_720_1280/0/1603203682939?e=1718841600&v=beta&t=qIKNoFXXY7Gsg5xfE2FNYd4odtS7tNalr5WvVQ069Xk" />

<h2 align="left"> Objetivo</h2>

<p>Este proyecto tiene como objetivo desarrollar un sistema de recomendación de videojuegos para usuarios de la plataforma Steam. El proyecto abarca desde el tratamiento de datos hasta la implementación de una API RESTful</p>

<div>
  <ul>
    <li><b>Tratamiento de datos:</b> El proyecto involucra la limpieza, preprocesamiento y análisis de un conjunto de datos masivo de videojuegos disponibles en Steam. Esto implica abordar aspectos como la eliminación de valores faltantes, la normalización de datos y la extracción de características relevantes.</li>
    <li><b>Implementación de algoritmos de recomendación:</b> Se utilizarán algoritmos de aprendizaje automático y técnicas de minería de datos para identificar patrones de comportamiento de los usuarios y recomendar videojuegos que se ajusten a sus preferencias.</li>
    <li><b>Desarrollo de una API RESTful:</b> Se creará una API RESTful que permita a los usuarios interactuar con el sistema de recomendación. La API proporcionará endpoints para obtener recomendaciones personalizadas, explorar el catálogo de videojuegos y consultar información detallada sobre cada juego.</li>
  </ul>
</div>
  </ul>
</div
><h2 align="left"> Configuración </h2>
<div>
  <ul>
    <li>Crea un entorno virtual</li>
    <li>Instalar las dependencias</li>
    
    pip install -r requirements.txt

<li>Ejecutar la aplicación:</li>

    uvicorn main:app --reload
  </ul>
</div>

<h2 align="left"> Endpoints de la API y Modelos de prediccion</h2>
<div>
  <ul>
    <li><b>Get("/developer/{desarrollador}")</b> Provides user-specific information including money spent, recommendation percentage, and item count.</li>
    <li><b>Get("/userdata/{User_id}"):</b>Identifies the user with the most playtime for a given genre and provides a list of accumulated playtime by release year.</li>
    <li><b>Get("/usergenre/{genre}"):</b>Retrieves the top 3 developers with the most user-recommended games for the specified year, considering positive reviews (reviews.recommend = True).</li>
    <li><b>Get("/best_developer_year/{año}"):</b>Analyzes user reviews for a specific developer and provides a dictionary with the total number of positive and negative reviews.</li>
    <li><b>Get("/developer_reviews_analysis/{developer}"):</b>This endpoint provides a detailed analysis of user reviews for a specific game developer</li>
    <li><b>Get("/recomendacion_juego/{Id_item}"):</b>This endpoint recommends games similar to a specific game based on machine learning algorithms</li>
    <li><b>Get("/recomendacion_usuario/{id_usuario}"):</b>This endpoint provides personalized game recommendations for a specific user based on their purchase history and preferences.</li>

  </ul>
</div>

<h2 align="left"> Análisis Exploratorio de Datos (EDA) </h2>
<p>se encuentra dentro del código principal del proyecto. Su objetivo principal es generar visualizaciones como gráficos y tablas para analizar y comprender mejor las relaciones y patrones existentes en los datos.</p>

<h2 align="left"> Modelo de Aprendizaje Automático </h2>
<p>El sistema de recomendación implementado en este proyecto utiliza la similitud del coseno para identificar juegos similares a un juego específico o para recomendar juegos a un usuario en particular. La similitud del coseno es una medida de similitud entre dos vectores que se basa en el ángulo entre ellos. En este contexto, los vectores representan las características de los juegos.</p>
