import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configuración de Experimentos en MLflow
mlflow.set_experiment("California_Housing_Benchmark")

def evaluar_modelo(y_true, y_pred):
    """Calcula las métricas estándar de regresión."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def graficar_residuos(y_true, y_pred, model_name):
    """Genera y guarda una gráfica de residuos."""
    residuos = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuos, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Gráfico de Residuos - {model_name}")
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    
    filename = f"residuos_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def ejecutar_entrenamiento():
    # 1. Carga de datos
    data_path = "datos/datos_limp/california_housing_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo {data_path}. Ejecuta datos_prep.py primero.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # ----------------------------------------------------
    # MODELO 1: Regresión Ridge (Baseline)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="Ridge_Regression_Baseline"):
        pipeline_ridge = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])
        
        param_grid_ridge = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
        grid_ridge = GridSearchCV(pipeline_ridge, param_grid_ridge, cv=kf, scoring='neg_root_mean_squared_error')
        grid_ridge.fit(X_train, y_train)
        
        best_ridge = grid_ridge.best_estimator_
        y_pred_ridge = best_ridge.predict(X_test)
        
        mae, rmse, r2 = evaluar_modelo(y_test, y_pred_ridge)
        
        # Registro en MLflow
        mlflow.log_params(grid_ridge.best_params_)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        
        fig_path = graficar_residuos(y_test, y_pred_ridge, "Ridge_Regression")
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
        
        mlflow.sklearn.log_model(best_ridge, "model_ridge")
        print(f"[Ridge] R2: {r2:.4f} | RMSE: {rmse:.4f}")

    # ----------------------------------------------------
    # MODELO 2: XGBoost Regressor
    # ----------------------------------------------------
    with mlflow.start_run(run_name="XGBoost_Optimized"):
        xgb = XGBRegressor(random_state=42)
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
        
        grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_xgb.fit(X_train, y_train)
        
        best_xgb = grid_xgb.best_estimator_
        y_pred_xgb = best_xgb.predict(X_test)
        
        mae, rmse, r2 = evaluar_modelo(y_test, y_pred_xgb)
        
        # Registro en MLflow
        mlflow.log_params(grid_xgb.best_params_)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        
        fig_path = graficar_residuos(y_test, y_pred_xgb, "XGBoost")
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
        
        mlflow.xgboost.log_model(best_xgb, "model_xgb")
        print(f"[XGBoost] R2: {r2:.4f} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    ejecutar_entrenamiento()