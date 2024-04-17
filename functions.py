import pandas as pd
import fastapi
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_steam = pd.read_parquet('data_steam.parquet')
data = data_steam.copy()
data_steam['release_year'] = data_steam['release_year'].astype(int)




''' 1) Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
       Number of items and percentage of Free content per year by developer company '''

def developer(desarrollador:str):
    data_steam['release_year'] = data_steam['release_year'].astype(int)

    '''Limpiar la cadena del desarrollador eliminando espacios adicionales y convirtiendo a minúsculas
    Clean the developer string by removing extra spaces and converting to lowercase'''
    desarrollador = desarrollador.strip().lower()

    # Filtrar por el desarrollador proporcionado / Filter by the provided developer
    developer_data = data_steam[data_steam['developer'].str.strip().str.lower() == desarrollador]

    # Agrupar por 'release_year' y calcular las métricas / Group by 'release_year' and calculate metrics
    result_data = developer_data.groupby('release_year').agg({
        'item_id': 'count',  # Cantidad total de ítems por año / Number of free items per year
        'price': lambda x: (x == 0).sum(),  # Cantidad de ítems gratuitos por año / Number of free items per year
    }).reset_index()

    # Calcular el porcentaje de contenido gratis / Calculate the percentage of free content
    result_data['free_content_percentage'] = round((result_data['price'] / result_data['item_id']) * 100, 2)

    # Seleccionar solo las columnas necesarias / Select only the necessary columns
    result_data = result_data[['release_year', 'item_id', 'free_content_percentage']]

    # Renombrar las columnas / Rename the columns
    result_data.columns = ['release_year', 'items_count', 'free_content_percentage']

    result_dict = {
        "Año / Year": result_data['release_year'].to_dict(),
        "Cantidad de items / Items quantity": result_data['items_count'].to_dict(),
        "Porcentaje_contenido_gratis": result_data['free_content_percentage'].to_dict()
    }
    return result_dict

''' 2) Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
       It should return the amount of money spent by the user, the recommendation percentage based on reviews.recommend and the number of items.'''

def userdata(User_id: str):

    ''' Limpiar la cadena del usuario eliminando espacios adicionales y convirtiendo a minúsculas
      Clean the user string by removing extra spaces and converting to lowercase'''
    
    User_id = User_id.strip().lower()

    # Filtrar por el usuario proporcionado / Filter by the provided user
    user_data = data_steam[data_steam['user_id'].str.strip().str.lower() == User_id]

    # Agrupar por 'user_id' y calcular las métricas / Group by 'user_id' and calculate metrics
    result_data = user_data.groupby('user_id').agg({
        'recommend': 'sum',
        'price': 'sum',  
        'item_id': 'count'  
    }).reset_index()

    # Calcular el porcentaje de recomendaciones / Calculate the recommendation percentage
    result_data['recommend_percentage'] = round((result_data['recommend'] / result_data['item_id']) * 100, 2)

    # Seleccionar solo las columnas necesarias / Select only the necessary columns
    result_data = result_data[['user_id', 'price', 'recommend_percentage','item_id']]

    # Renombrar las columnas / Rename the columns
    result_data.columns = ['user', 'money_spent', 'recommend_percentage','item_id']

    # Obtener los valores como un diccionario / Get the values as a dictionary
    user_dict = result_data.to_dict(orient='records')[0]

    # Formatear el resultado según el ejemplo / Format the result according to the example
    formatted_result = {
        "user": user_dict['user'],
        "money_spent": f"{user_dict['money_spent']} USD",
        "Porcentaje de recomendaciones": f"{user_dict['recommend_percentage']}%",
        "Cantidad de ítems": user_dict['item_id']
    }
    return formatted_result

''' 3) Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
       It should return the user who accumulates the most hours played for the given genre and a list of the accumulation of hours played by year of release '''

def UserForGenre(genre: str):
    ''' Limpiar la cadena del género eliminando espacios adicionales y convirtiendo a minúsculas
    Clean the genre string by removing extra spaces and converting to lowercase'''
    genre = genre.strip().lower()

    # Filtrar por el género proporcionado / Filter by the provided genre
    genre_data = data_steam[data_steam['genre'].str.strip().str.lower() == genre]

    # Encontrar el usuario con más horas jugadas para el género dado / Find the user with the most playtime for the given genre
    
    user_entry = genre_data.loc[genre_data['playtime_forever'].idxmax(), ['user_id', 'playtime_forever']]

    # Agrupar por 'release_year' y calcular la acumulación de horas jugadas por año / Group by 'release_year' and calculate the accumulation of playtime per year
    playtime_by_year = genre_data.groupby('release_year')['playtime_forever'].sum().reset_index()

    # Crear un diccionario para las horas jugadas por año / Create a dictionary for the playtime per year
    hours_by_year = [{"Año": year, "Horas": hours} for year, hours in zip(playtime_by_year['release_year'], playtime_by_year['playtime_forever'])]

    # Crear el resultado final / Create the final result
    result = {
        "Usuario con más horas jugadas para Género": user_entry['user_id'],
        "Horas jugadas": hours_by_year
    }

    return result

''' 4) Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
       Return the top 3 developers with the MOST user-recommended games for the given year. (reviews.recommend = True and positive reviews)'''

def best_developer_year(year: int):
    data_steam['release_year'] = data_steam['release_year'].astype(int)
    data_1 = data_steam[['release_year', 'recommend', 'developer', 'sentiment_analysis']]
    
    # Filtrar los juegos por año y por recomendación positiva / Filter games by year and positive recommendation
    df_year = data_1[(data_1["release_year"] == year) & (data_1["recommend"] == True) & (data_1["sentiment_analysis"] == 2)]

    if df_year.empty:
        return {"error": "No hay datos para ese año."}

    ''' Contar el número de juegos recomendados por desarrollador y devolver los tres primeros desarrolladores
        Count the number of recommended games by developer and return the top three developers '''
    best_developer = df_year["developer"].value_counts().head(3).index.tolist()

    # Devolver el top 3 de desarrolladores / devolver top 3_desarrolladores
    return {"Puesto 1": best_developer[0], "Puesto 2": best_developer[1], "Puesto 3": best_developer[2]}


''' 5) Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como
      llave y una lista con la cantidad total de registros de reseñas de usuarios que se 
      encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. 
      
      According to the developer, a dictionary is returned with the developer name as the key and a
      list with the total number of user review records that are categorized with a sentiment analysis 
      as a positive or negative value.'''

def developer_reviews_analysis(developer: str):
    developer_data = data_steam[(data_steam['developer'] == developer) &
                                data_steam['sentiment_analysis'].isin([0, 2])]
    developer_data = developer_data.groupby(['developer', 'sentiment_analysis']).size().reset_index(name='count')

    # Crear un diccionario con el formato deseado / Create a dictionary with the desired format
    result_dict = {developer: {'Negative': 0, 'Positive': 0}}

    for _, row in developer_data.iterrows():
        sentiment_label = 'Negative' if row['sentiment_analysis'] == 0 else 'Positive'
        result_dict[developer][sentiment_label] = row['count']
    return result_dict

''' 6) Machine Learning Ingresando el id de producto, deberíamos recibir una lista con 5 
       juegos recomendados similares al ingresado.
       
       Machine Learning: By entering the product id, we should receive a list of 5
       recommended games similar to the one entered'''

def recomendacion_juego(Id_item):
    try: 
        Id_item = str(Id_item)
        # se carga los datasets que se va a utilizar para dos dataframes distintos / Load the datasets that will be used for two different dataframes
        data1 = pd.read_parquet("ML/steam_games.parquet")
        
        data_games_steam = pd.read_parquet("ML/steam_id.parquet")

        # crear una matriz de características de los juegos / Creating a Feature Matrix for Games
        tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r"\b[a-zA-Z0-9]\w+\b")
        data_vector = tfidv.fit_transform(data1["specs"])

        data_vector_df = pd.DataFrame(data_vector.toarray(), index=data1["item_id"], columns = tfidv.get_feature_names_out())

        # calcular la similitud coseno entre los juegos en la matriz de características / Calculate the cosine similarity between the games in the feature matrix
        vector_similitud_coseno = cosine_similarity(data_vector_df.values)

        cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)

        juego_simil = cos_sim_df.loc[Id_item]

        simil_ordenada = juego_simil.sort_values(ascending=False)
        resultado = simil_ordenada.head(6).reset_index()

        result_df = resultado.merge(data_games_steam, on="item_id",how="left")

        # La función devuelve una lista de los 6 juegos más similares al juego dado / This function takes an item id as input and returns a list of the 6 most similar games based on cosine similarity
        juego_title = data_games_steam[data_games_steam["item_id"] == Id_item]["title"].values[0]

        # mensaje que indica el juego original y los juegos recomendados / Message indicating the original game and recommended games
        mensaje = f"Si te gustó el juego {Id_item} : {juego_title}, también te pueden gustar:"

        result_dict = {
            "mensaje": mensaje,
            "juegos recomendados": result_df["title"][1:6].tolist()
        }
        return result_dict
        
    except Exception as e:
        return {'error': str(e)}
    
''' 7) Machine Learning se ingresa el Id_usuario y se le recomienda 5 juegos al mismo / 
       Machine Learning ingresa el user_id y recomienda 5 juegos al mismo'''

def recomendacion_usuario(id_usuario):
    try:
    #Seleccionar características (X) y la etiqueta (y) / Select features (X) and the label (y)
        X = data_steam[["item_id", "release_year", "price", "sentiment_analysis", "playtime_forever"]]
        y = data_steam["recommend"]

        #Divide el conjunto de datos en conjuntos de entrenamiento y prueba / Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Crea y entrena el modelo / Create and train the model
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        #------------------------------------------------------------------------------------------------------
        
        # Verifica si el usuario existe en el conjunto de datos / Check if the user exists in the dataset
        if id_usuario not in data_steam["user_id"].values:
            print(f"El usuario {id_usuario} no existe en el conjunto de datos.")
            return None

        # Filtra el conjunto de datos para obtener las características de juegos no etiquetados para el usuario / Filter the dataset to get the features of unlabeled games for the user
        juegos_sin_etiqueta = data_steam[data_steam["user_id"] == id_usuario][["item_id", "release_year", "price", "sentiment_analysis", "playtime_forever"]]

        # Asegúrate de que haya al menos un juego sin etiquetar para el usuario / Make sure there is at least one unlabeled game for the user
        if juegos_sin_etiqueta.empty:
            print(f"No hay juegos sin etiquetar para el usuario {id_usuario}.")
            return None

        # Utiliza el modelo entrenado para predecir las preferencias del usuario para los juegos sin etiquetar / Use the trained model to predict the user's preferences for the unlabeled games
        preferencias_usuario = modelo.predict(juegos_sin_etiqueta)

        # Combina las predicciones con la información del juego y selecciona los 5 mejores / Combine predictions with game information and select the top 5
        juegos_sin_etiqueta["recommend"] = preferencias_usuario
        juegos_recomendados = juegos_sin_etiqueta.sort_values(by="recommend", ascending=False).head(5)

        # Realiza una fusión con el conjunto de datos original para obtener el nombre del juego / Merge with the original dataset to get the game name
        juegos_recomendados = pd.merge(juegos_recomendados, data_steam[["item_id", "title"]], on="item_id", how="left")

        # Elimina duplicados basados en "title" / Remove duplicates based on "title"
        juegos_recomendados = juegos_recomendados.drop_duplicates(subset="title")

        # Reinicia el índice y luego incrementa en 1 / Reset the index and then increment by 1
        juegos_recomendados.reset_index(drop=True, inplace=True)
        juegos_recomendados.index += 1
        
        # Suponiendo que "juegos_recomendados" es un DataFrame con las columnas mencionadas / Assuming "recommended_games" is a DataFrame with the mentioned columns
        datos_dict = juegos_recomendados[["title", "release_year", "price", "sentiment_analysis", "recommend", "playtime_forever"]].to_dict(orient="records")

        return datos_dict
        
    except Exception as e:
        return {'error': str(e)}

