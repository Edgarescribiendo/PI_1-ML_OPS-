import fastapi
import pandas as pd
from functions import *
from fastapi import FastAPI
from typing import List,Dict,Tuple,Sequence,Callable, Optional,Any,Union
from typing import Union
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

#data_steam = pd.read_parquet('data_steam.parquet')

@app.get("/")
#http://127.0.0.1:8000 Ruta madre del puerto

def root():
    return {'message':'Welcome!'}

''' 1) Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. '''

@app.get("/developer/{desarrollador}")
async def desarrollador(desarrollador:str):
    try:
        resultado = developer(desarrollador)
        return resultado
    except Exception as e:
        return {"error": str(e)}
    
''' 2) Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación 
    en base a reviews.recommend y cantidad de items.'''

@app.get("/userdata/{User_id}")
def user(User_id:str):
    try:
        result = userdata(User_id)
        return result
    except Exception as e:
        return {'error': str(e)}    

''' 3) Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista
    de la acumulación de horas jugadas por año de lanzamiento. '''

@app.get("/usergenre/{genre}")
def genre(genre: str):
    try:
        result = UserForGenre(genre)
        return result
    except Exception as e:
        return {'error': str(e)}  
    
''' 4) Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el 
    año dado. (reviews.recommend = True y comentarios positivos)'''

@app.get("/best_developer_year/{año}")
async def Best_developer_year(year: int):
    try:
        result2 = best_developer_year(year)
        return result2
    except Exception as e:
        return {"error": str(e)}   


''' 5) Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como
      llave y una lista con la cantidad total de registros de reseñas de usuarios que se 
      encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. '''

@app.get("/developer_reviews_analysis/{developer}")
def dev_reviews_analysis(developer: str):
    try:
        result = developer_reviews_analysis(developer)
        return result
    except Exception as e:
        return {'error': str(e)}  
    
'''  6ta función - Machine Learning Ingresando el id de producto, deberíamos recibir una lista con 5 
    juegos recomendados similares al ingresado.'''
@app.get("/recomendacion_juego/{Id_item}")
def get_recomedacion_juego(Id_item: int):
    try:
        result = recomendacion_juego((Id_item))
        return result
    except Exception as e:
        return {'error': str(e)}  


@app.get("/recomendacion_usuario/{id_usuario}")
def get_recomedacion_usuario (id_usuario: str):
    try:
        result= recomendacion_usuario(id_usuario)
        return result
    except Exception as e:
        return {'error': str(e)}  


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)