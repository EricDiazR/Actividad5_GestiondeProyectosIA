import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def cargar_datos_iniciales(ruta_destino: str) -> pd.DataFrame:
    """Carga el dataset público de California Housing y lo guarda en datos_ini."""
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    df.to_csv(ruta_destino, index=False)
    print(f"[+] Datos iniciales guardados en: {ruta_destino}")
    return df

def limpiar_y_transformar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de negocio, imputación y eliminación de valores atípicos."""
    df_limpio = df.copy()
    
    # 1. Tratamiento de nulos (si existieran)
    df_limpio = df_limpio.dropna()
    
    # 2. Reglas de negocio / Filtrado de atípicos extremos
    # Casas con más de 10 habitaciones promedio o precios topeados (> 5.0)
    df_limpio = df_limpio[df_limpio['AveRooms'] < 10]
    df_limpio = df_limpio[df_limpio['MedHouseVal'] < 5.0]
    
    # 3. Creación de nuevas variables (Feature Engineering)
    df_limpio['BedroomsPerRoom'] = df_limpio['AveBedrms'] / df_limpio['AveRooms']
    df_limpio['PopulationPerHouse'] = df_limpio['Population'] / df_limpio['AveOccup']
    
    return df_limpio

def guardar_datos_limpios(df: pd.DataFrame, ruta_destino: str) -> None:
    """Guarda el DataFrame procesado en la carpeta de datos limpios."""
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    df.to_csv(ruta_destino, index=False)
    print(f"[+] Datos limpios guardados en: {ruta_destino}")

if __name__ == "__main__":
    PATH_INI = "datos/datos_ini/california_housing_raw.csv"
    PATH_LIMP = "datos/datos_limp/california_housing_clean.csv"
    
    raw_df = cargar_datos_iniciales(PATH_INI)
    clean_df = limpiar_y_transformar(raw_df)
    guardar_datos_limpios(clean_df, PATH_LIMP)