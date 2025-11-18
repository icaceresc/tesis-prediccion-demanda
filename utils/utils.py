# -*- coding: utf-8 -*-
"""
Módulo de Utilidades (utils.py) - v20.1 (Limpio)
Contiene toda la maquinaria para carga, filtrado, modelado y visualización.
"""

import pandas as pd
from dbfread import DBF
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pmdarima as pm 
import warnings 
import logging
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit 
from tqdm.notebook import tqdm

# =============================================================================
# --- SECCIÓN 1: CARGA Y LIMPIEZA INICIAL DE DATOS (NB 01) ---
# =============================================================================

def cargar_datos_dbf(ruta_directorio_raw):
    """Carga las tablas VENTA, DVENTA e IN-F01 desde archivos DBF."""
    try:
        tabla_venta = DBF(os.path.join(ruta_directorio_raw, 'VENTA.DBF'), encoding='latin-1', load=True)
        venta_df = pd.DataFrame(iter(tabla_venta))
        tabla_dventa = DBF(os.path.join(ruta_directorio_raw, 'DVENTA.DBF'), encoding='latin-1', load=True)
        dventa_df = pd.DataFrame(iter(tabla_dventa))
        tabla_productos = DBF(os.path.join(ruta_directorio_raw, 'in-f01.DBF'), encoding='latin-1', load=True)
        productos_df = pd.DataFrame(iter(tabla_productos))
        print("Datos brutos cargados correctamente.")
        return venta_df, dventa_df, productos_df
    except Exception as e:
        print(f"Error al cargar los archivos DBF: {e}")
        return None, None, None

def integrar_y_limpiar(venta_df, dventa_df, productos_df):
    """Integra las tablas, añade el monto de venta y realiza una limpieza básica."""
    venta_df = venta_df[['ID', 'FECHA_VEN']].rename(columns={'ID': 'id_venta', 'FECHA_VEN': 'fecha'})
    dventa_df = dventa_df[['IDVENTA', 'CODI', 'CANTIDAD', 'PRE_VEN']].rename(columns={'IDVENTA': 'id_venta', 'CODI': 'codigo_producto', 'CANTIDAD': 'cantidad', 'PRE_VEN': 'precio_venta'})
    productos_df = productos_df[['CODI', 'DES']].rename(columns={'CODI': 'codigo_producto', 'DES': 'descripcion_producto'})
    ventas_full_df = pd.merge(dventa_df, venta_df, on='id_venta', how='left')
    ventas_full_df['fecha'] = pd.to_datetime(ventas_full_df['fecha'], errors='coerce')
    ventas_full_df.dropna(subset=['fecha'], inplace=True)
    ventas_full_df['cantidad'] = pd.to_numeric(ventas_full_df['cantidad'], errors='coerce').fillna(0)
    ventas_full_df['precio_venta'] = pd.to_numeric(ventas_full_df['precio_venta'], errors='coerce').fillna(0)
    ventas_full_df.dropna(subset=['codigo_producto'], inplace=True)
    ventas_full_df['monto_venta'] = ventas_full_df['cantidad'] * ventas_full_df['precio_venta']
    ventas_full_df['codigo_producto'] = ventas_full_df['codigo_producto'].str.strip()
    productos_df['codigo_producto'] = productos_df['codigo_producto'].str.strip()
    ventas_full_df = pd.merge(ventas_full_df, productos_df, on='codigo_producto', how='left')
    print("Tablas integradas y limpiadas (incluyendo monto de venta).")
    return ventas_full_df

def agregar_ventas_mensuales(df):
    """Agrega las ventas a un formato de serie de tiempo mensual."""
    df['mes'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
    ventas_mensuales_df = df.groupby(
        ['codigo_producto', 'descripcion_producto', 'mes']
    ).agg(
        cantidad=('cantidad', 'sum'),
        monto_venta=('monto_venta', 'sum') 
    ).reset_index()
    ventas_mensuales_df = ventas_mensuales_df.sort_values(by=['codigo_producto', 'mes'])
    if not ventas_mensuales_df.empty:
        min_year = ventas_mensuales_df['mes'].dt.year.min()
        ventas_mensuales_df['FECHA_MES_NRO'] = (ventas_mensuales_df['mes'].dt.year - min_year) * 12 + ventas_mensuales_df['mes'].dt.month
    else:
        ventas_mensuales_df['FECHA_MES_NRO'] = None
    print("Ventas agregadas a nivel mensual para cantidad y monto (con FECHA_MES_NRO).")
    return ventas_mensuales_df


# =============================================================================
# --- SECCIÓN 2: FILTRADO DE DATOS (NB 02) ---
# =============================================================================

def filtrar_por_relevancia(df, n_anios=4):
    """Filtra productos que no tienen ventas en cada uno de los últimos N años."""
    if 'mes' not in df.columns or df.empty: return df
    df_temp = df.copy()
    df_temp['anio'] = df_temp['mes'].dt.year
    ultimo_anio = df_temp['anio'].max()
    if pd.isna(ultimo_anio): return pd.DataFrame(columns=df.columns)
    anios_relevantes = range(int(ultimo_anio) - n_anios + 1, int(ultimo_anio) + 1)
    articulos_validos = set(df_temp['codigo_producto'].unique())
    for anio in anios_relevantes:
        articulos_en_anio = set(df_temp[df_temp['anio'] == anio]['codigo_producto'].unique())
        articulos_validos &= articulos_en_anio
    print(f"Filtrando por relevancia: {len(articulos_validos)} productos vendidos en cada uno de los últimos {n_anios} años.")
    return df[df['codigo_producto'].isin(articulos_validos)]

def filtrar_por_volumen_y_dispersion(df, col_producto='codigo_producto', col_fecha='mes', col_valor='cantidad', min_meses_venta=12, quantile_filtro=0.25):
    """Aplica filtros de relevancia (dispersión y volumen) a un dataframe."""
    if df.empty: return df
    n_productos_inicial = df[col_producto].nunique()
    ventas_por_producto = df[df[col_valor] > 0].groupby(col_producto)[col_fecha].count()
    productos_activos = ventas_por_producto[ventas_por_producto >= min_meses_venta].index
    df_filtrado_1 = df[df[col_producto].isin(productos_activos)]
    n_prod_post_dispersion = df_filtrado_1[col_producto].nunique()
    if n_prod_post_dispersion == 0: return df_filtrado_1
    volumen_total = df_filtrado_1.groupby(col_producto)[col_valor].sum()
    umbral_volumen = volumen_total.quantile(quantile_filtro)
    productos_relevantes = volumen_total[volumen_total > umbral_volumen].index
    df_filtrado_final = df_filtrado_1[df_filtrado_1[col_producto].isin(productos_relevantes)]
    n_prod_final = df_filtrado_final[col_producto].nunique()
    print(f"¡Filtro de volumen y dispersión completado! Productos reducidos de {n_productos_inicial} a {n_prod_final}.")
    return df_filtrado_final

def filtrar_outliers_iqr(df, value_col='cantidad', group_by_col=None):
    """Filtra outliers usando IQR, opcionalmente agrupando."""
    if df.empty: return df
    if group_by_col:
        Q1 = df.groupby(group_by_col)[value_col].transform('quantile', 0.25)
        Q3 = df.groupby(group_by_col)[value_col].transform('quantile', 0.75)
    else:
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    lim_inf = np.where(IQR == 0, df[value_col], lim_inf)
    lim_sup = np.where(IQR == 0, df[value_col], lim_sup)
    df_sin_outliers = df[(df[value_col] >= lim_inf) & (df[value_col] <= lim_sup)]
    filas_eliminadas = len(df) - len(df_sin_outliers)
    metodo = "por producto" if group_by_col else "global"
    print(f"Filtro IQR ({metodo}) aplicado a '{value_col}'. Se eliminaron {filas_eliminadas} filas.")
    return df_sin_outliers


# =============================================================================
# --- SECCIÓN 3: MODELADO Y EVALUACIÓN (NB 03) ---
# =============================================================================

def mase_score(y_true, y_pred, y_train, m=12):
    """Calcula el Mean Absolute Scaled Error (MASE) robusto."""
    try:
        y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
        mae_test = mean_absolute_error(y_true, y_pred)
        if len(y_train) <= m:
            mae_train_naive = np.mean(np.abs(np.diff(y_train)))
        else:
            mae_train_naive = np.mean(np.abs(y_train[m:] - y_train[:-m]))
        if mae_train_naive < 1e-10: 
            return np.nan 
        return mae_test / mae_train_naive
    except Exception:
        return np.nan

def calcular_metricas(y_true, y_pred, y_train, metrics_list, m=12):
    """Calcula un diccionario de métricas. y_train es necesario para MASE."""
    metricas = {}
    y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
    if len(y_true) == 0: return {metric_name: np.nan for metric_name in metrics_list}
    if 'R2' in metrics_list: metricas['R2'] = r2_score(y_true, y_pred)
    if 'MAE' in metrics_list: metricas['MAE'] = mean_absolute_error(y_true, y_pred)
    if 'RMSE' in metrics_list: metricas['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    if 'ME' in metrics_list: metricas['ME'] = np.mean(y_true - y_pred)
    if 'MASE' in metrics_list: metricas['MASE'] = mase_score(y_true, y_pred, y_train, m=m)
    return metricas


def ejecutar_experimentos(datasets_dict, df_info_productos, target_column, models_to_run, 
                          metrics_to_calculate, maxiter_sarima=50, n_splits_cv=3, test_size_cv=12):
    """Ejecuta los experimentos de modelado usando TimeSeries Cross-Validation."""
    resultados_completos = {}
    metric_cols_base = [f"{metric}_{model}" for model in models_to_run for metric in metrics_to_calculate]
    winner_cols = [f"{metric}_ganador" for metric in metrics_to_calculate]
    winner_model_cols = [f"modelo_ganador_{metric}" for metric in metrics_to_calculate]

    for nombre_dataset, df in datasets_dict.items():
        logging.info(f"Procesando dataset: {nombre_dataset} para target: {target_column}")
        if df.empty:
            logging.warning(f"Dataset {nombre_dataset} está vacío. Saltando...")
            continue
            
        productos_agrupados = df.groupby('codigo_producto') 
        lista_resultados_productos = []
        
        for codigo_producto, data_producto in tqdm(productos_agrupados, desc=f"Modelando {nombre_dataset}"):
            data_producto = data_producto.sort_values(by='mes').reset_index(drop=True) 
            if 'FECHA_MES_NRO' not in data_producto.columns:
                logging.error(f"Falta 'FECHA_MES_NRO' en {nombre_dataset} para producto {codigo_producto}. Saltando.")
                continue

            X = data_producto[['FECHA_MES_NRO']] 
            y = data_producto[target_column]
            n_obs = len(y)
            
            # --- Configuración de CV (TimeSeriesSplit) ---
            max_splits_posibles = (n_obs // test_size_cv) - 1
            
            if max_splits_posibles < 2: 
                logging.warning(f"Producto {codigo_producto} omitido. Datos ({n_obs}) insuficientes para n_splits >= 2 (max_splits_posibles: {max_splits_posibles}).")
                continue 

            n_splits_final = min(n_splits_cv, max_splits_posibles)
            tscv = TimeSeriesSplit(n_splits=n_splits_final, test_size=test_size_cv)
            
            n_train_min = n_obs - (n_splits_final * test_size_cv)
            k_knn = max(1, min(3, n_train_min - 1)) 
            
            modelos = {
                'Lineal': LinearRegression(),
                'Cuadrático': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                'KNN': KNeighborsRegressor(n_neighbors=k_knn)
            }
            metricas_folds = {f'{m}_{met}': [] for m in models_to_run for met in metrics_to_calculate}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # --- 3a. Modelos Scikit-learn ---
                    for model_name in models_to_run:
                        if model_name in modelos:
                            try:
                                modelo = modelos[model_name]
                                if model_name == 'KNN':
                                    k_fold = max(1, min(k_knn, len(y_train) - 1))
                                    modelo.set_params(n_neighbors=k_fold)
                                modelo.fit(X_train, y_train)
                                y_pred = modelo.predict(X_test)
                                metricas_oos = calcular_metricas(y_test, y_pred, y_train, metrics_to_calculate, m=12)
                                for metric_name, value in metricas_oos.items():
                                    metricas_folds[f'{model_name}_{metric_name}'].append(value)
                            except Exception as e:
                                for metric_name in metrics_to_calculate:
                                    metricas_folds[f'{model_name}_{metric_name}'].append(np.nan)
                    
                    # --- 3b. Modelo SARIMA (pmdarima) ---
                    if 'SARIMA' in models_to_run:
                        try:
                            X_train_sarima = X_train.values.reshape(-1, 1)
                            X_test_sarima = X_test.values.reshape(-1, 1)
                            modelo_sarima = pm.auto_arima(
                                y_train.values, X=X_train_sarima, m=12, seasonal=True,
                                stepwise=True, suppress_warnings=True, error_action='ignore',
                                maxiter=maxiter_sarima
                            )
                            y_pred_sarima = modelo_sarima.predict(n_periods=len(y_test), X=X_test_sarima)
                            metricas_oos_sarima = calcular_metricas(y_test, y_pred_sarima, y_train, metrics_to_calculate, m=12)
                            for metric_name, value in metricas_oos_sarima.items():
                                metricas_folds[f'SARIMA_{metric_name}'].append(value)
                        except Exception as e:
                            for metric_name in metrics_to_calculate:
                                metricas_folds[f'SARIMA_{metric_name}'].append(np.nan)

            # --- 4. Promediar Métricas (Post-CV) ---
            resultados_producto = {'codigo_producto': codigo_producto, 'N_OBS': n_obs, 'N_SPLITS_CV': n_splits_final, 'K_KNN_USADO': k_knn}
            for model_name in models_to_run:
                for metric_name in metrics_to_calculate:
                    col_name = f'{model_name}_{metric_name}_CV_mean'
                    metric_values = metricas_folds[f'{model_name}_{metric_name}']
                    resultados_producto[col_name] = np.nanmean(metric_values) if metric_values else np.nan
            for metric_col in metric_cols_base:
                col_name_cv = f'{metric_col}_CV_mean'
                if col_name_cv not in resultados_producto:
                    resultados_producto[col_name_cv] = np.nan

            # --- 5. Determinar Ganadores ---
            for metric_name in metrics_to_calculate:
                is_max_better = metric_name in ['R2']
                is_min_better = metric_name in ['MAE', 'RMSE', 'MASE']
                best_model_name, best_model_value = None, None
                if metric_name == 'ME': best_model_value = np.inf
                else: best_model_value = -np.inf if is_max_better else np.inf

                for model_name in models_to_run:
                    col_name = f'{model_name}_{metric_name}_CV_mean'
                    if col_name in resultados_producto:
                        current_value = resultados_producto[col_name]
                        if pd.isna(current_value): continue
                        if is_max_better and current_value > best_model_value:
                            best_model_value, best_model_name = current_value, model_name
                        elif is_min_better and current_value < best_model_value:
                            best_model_value, best_model_name = current_value, model_name
                        elif metric_name == 'ME' and np.abs(current_value) < np.abs(best_model_value):
                            best_model_value, best_model_name = current_value, model_name

                resultados_producto[f'modelo_ganador_{metric_name}'] = best_model_name
                resultados_producto[f'{metric_name}_ganador'] = best_model_value if best_model_name is not None and not np.isinf(best_model_value) else np.nan
            
            all_winner_cols = winner_model_cols + winner_cols
            for col in all_winner_cols:
                if col not in resultados_producto:
                    resultados_producto[col] = np.nan
            lista_resultados_productos.append(resultados_producto)

        # --- 6. Crear DataFrame de Resultados ---
        if lista_resultados_productos:
            df_resultados = pd.DataFrame(lista_resultados_productos)
            df_resultados = df_resultados.merge(
                df_info_productos.astype({'codigo_producto': str}), on='codigo_producto', how='left'
            )
            metric_cols_base_cv = [f"{col}_CV_mean" for col in metric_cols_base]
            ordered_cols = ['codigo_producto', 'descripcion_producto', 'N_OBS', 'N_SPLITS_CV', 'K_KNN_USADO'] + \
                           sorted(winner_model_cols) + sorted(winner_cols) + sorted(metric_cols_base_cv)
            ordered_cols = [col for col in ordered_cols if col in df_resultados.columns] 
            df_resultados = df_resultados[ordered_cols]
            resultados_completos[nombre_dataset] = df_resultados
            print(f"Resultados (con CV) generados para {len(df_resultados)} productos.")
        else:
            print(f"No se generaron resultados para {nombre_dataset}.")
            resultados_completos[nombre_dataset] = pd.DataFrame()
        logging.info(f"Procesamiento de {nombre_dataset} completado.")
    logging.info("¡Todos los experimentos de modelado (con CV) han finalizado!")
    return resultados_completos


# =============================================================================
# --- ¡NUEVA SECCIÓN 4! FUNCIONES HELPER (NB 04) ---
# (Movidas desde el notebook a utils.py para mantener el código ligero)
# =============================================================================

def load_cv_results(input_dir, datasets_list, target_name):
    """
    Carga todos los archivos de resultados de CV para un target específico.
    """
    results_dict = {}
    logging.info(f"Iniciando carga de resultados para el target: {target_name.upper()}")
    
    target_input_dir = os.path.join(input_dir, target_name)
    
    for dataset_name in datasets_list:
        file_path = os.path.join(target_input_dir, f'resultados_detallados_{dataset_name}_{target_name}_CV.csv')
        try:
            df = pd.read_csv(file_path, dtype={'codigo_producto': str})
            results_dict[dataset_name] = df
            logging.info(f" - Archivo cargado: {os.path.basename(file_path)} (Productos: {len(df)})")
        except FileNotFoundError:
            logging.warning(f" - Archivo NO encontrado, omitiendo: {os.path.basename(file_path)}")
    
    if not results_dict:
        logging.error(f"No se pudo cargar ningún archivo de resultados para {target_name}. Verifica la ruta: {target_input_dir}")
        
    return results_dict


def load_universo_relevante_counts(processed_data_dir, datasets_list):
    """
    Carga los archivos de *datos* (ej. ventas_N_48_P.csv) para obtener el conteo
    del "Universo Relevante" (ej. 2115).
    """
    counts = {}
    
    # Necesitamos el conteo de F_48_NP y N_48_P para la lógica del gráfico
    # Usamos 'cantidad' como directorio base, ya que los conteos de productos son iguales
    base_data_dir = os.path.join(processed_data_dir, 'cantidad')
    
    # 1. Cargar el dataset F_48_NP (el que fue modelado)
    dataset_name = datasets_list[0] # Ej: 'F_48_NP'
    path_f48np = os.path.join(base_data_dir, f'ventas_{dataset_name}.csv')
    try:
        df_f48np = pd.read_csv(path_f48np, dtype={'codigo_producto': str})
        counts['total_modelados_cv'] = df_f48np['codigo_producto'].nunique() # N=1044
        logging.info(f"Conteo 'total_modelados_cv' (de {dataset_name}): {counts['total_modelados_cv']}")
    except FileNotFoundError:
        logging.error(f"No se encontró {path_f48np} para contar 'total_modelados_cv'.")
        counts['total_modelados_cv'] = 0

    # 2. Cargar el dataset N_48_P (el universo relevante ANTES del filtro IQR)
    # Este archivo fue guardado por el Notebook 02
    path_n48p = os.path.join(base_data_dir, 'ventas_N_48_P.csv')
    try:
        df_n48p = pd.read_csv(path_n48p, dtype={'codigo_producto': str})
        counts['total_relevante'] = df_n48p['codigo_producto'].nunique() # N=2115
        logging.info(f"Conteo 'total_relevante' (de N_48_P): {counts['total_relevante']}")
    except FileNotFoundError:
        logging.error(f"No se encontró ventas_N_48_P.csv para contar 'total_relevante'.")
        counts['total_relevante'] = counts.get('total_modelados_cv', 0) # Fallback
        
    return counts


def get_total_product_universe(raw_data_dir):
    """
    Carga el archivo 'in-f01.DBF' para obtener el conteo total (N=11596).
    """
    try:
        path = os.path.join(raw_data_dir, 'in-f01.DBF')
        tabla_productos = DBF(path, encoding='latin-1', load=True)
        df = pd.DataFrame(iter(tabla_productos))
        total_count = df['CODI'].nunique()
        logging.info(f"Universo total de productos (de {os.path.basename(path)}): {total_count}")
        return total_count
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo 'in-f01.DBF' en {raw_data_dir}.")
        return 0
    except Exception as e:
        logging.error(f"Error al cargar el universo total de productos (in-f01.DBF): {e}")
        return 0


# =============================================================================
# --- SECCIÓN 5: ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS (NB 04) ---
# (Funciones de v18/v19)
# =============================================================================

def generar_resumenes_cv(resultados_completos, main_metric='MASE', threshold=1.0, comparison_op='less_than'):
    """
    Genera tablas resumen genéricas basadas en una métrica y umbral de CV.
    (Función sin cambios)
    """
    resumen_total = []
    resumen_filtrado = []

    winning_col = f'modelo_ganador_{main_metric}'
    value_col = f'{main_metric}_ganador'

    for nombre, df_res in resultados_completos.items():
        if df_res.empty or winning_col not in df_res.columns:
            logging.warning(f"Columnas requeridas no encontradas en {nombre}. Saltando resumen.")
            continue
        
        df_res_valid = df_res.dropna(subset=[winning_col, value_col])

        if df_res_valid.empty:
            logging.warning(f"No hay datos válidos (sin NaN) para resumir en {nombre}.")
            continue
            
        # --- Resumen Total ---
        conteo_total = df_res_valid[winning_col].value_counts().reset_index()
        conteo_total.columns = ['Modelo', nombre]
        resumen_total.append(conteo_total.set_index('Modelo'))
        
        # --- Resumen Filtrado (basado en 'comparison_op') ---
        if comparison_op == 'less_than':
            df_filtrado_metric = df_res_valid[df_res_valid[value_col] < threshold]
        elif comparison_op == 'greater_than':
            df_filtrado_metric = df_res_valid[df_res_valid[value_col] > threshold]
        else:
            df_filtrado_metric = pd.DataFrame(columns=df_res_valid.columns) # Vacío
            
        if not df_filtrado_metric.empty:
            conteo_filtrado = df_filtrado_metric[winning_col].value_counts().reset_index()
            conteo_filtrado.columns = ['Modelo', nombre]
            resumen_filtrado.append(conteo_filtrado.set_index('Modelo'))

    df_resumen_total = pd.concat(resumen_total, axis=1).fillna(0).astype(int) if resumen_total else pd.DataFrame()
    df_resumen_filtrado = pd.concat(resumen_filtrado, axis=1).fillna(0).astype(int) if resumen_filtrado else pd.DataFrame()
    
    print(f"Tablas resumen generadas usando '{main_metric}' como métrica principal.")
    
    return df_resumen_total, df_resumen_filtrado

def _plot_pie_base(sizes, labels, colors, explode, title_text, 
                   ax=None, save_path_prefix=None, save_formats=['png', 'eps'], 
                   global_font_size=16, fig_size=(7, 7)): 
    """
    Función base interna para crear los gráficos de torta estilo tesis (v22).
    - CAMBIO v24.3: Tamaño de fuente de la caja (Universo) ahora es relativo (global_font_size * 0.8)
    - CAMBIO: Texto interno siempre negro.
    - CAMBIO: Caja de total movida a la esquina SUPERIOR derecha.
    """
    
    _ax_was_none = ax is None
    if _ax_was_none:
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": global_font_size, "text.color": "black",
            "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"
        })
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    total = sum(sizes)
    if total == 0:
        logging.warning("El total de datos para el gráfico de torta es 0.")
        ax.text(0.5, 0.5, "Sin datos", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if _ax_was_none: plt.close(fig)
        return ax
        
    percentages = [(s / total) * 100 for s in sizes]

    # --- Función Autopct (INTERNA) ---
    def func_autopct_internal(pct):
        absolute = int(round(pct/100.*total))
        return f"{pct:.1f}%\n(N={absolute})" 

    # --- 4. Dibujar el Gráfico de Torta ---
    wedges, texts_int, autotexts_int = ax.pie(
        sizes, 
        explode=explode,
        labels=None, 
        autopct=func_autopct_internal,
        colors=colors, 
        shadow=False, 
        startangle=90, 
        textprops=dict(color="black", fontsize=global_font_size, weight="normal"), # <-- Usa global_font_size
        pctdistance=0.75, 
        labeldistance=1.1 
    )
    
    # --- 5. Configurar Etiquetas Internas ---
    for i, at in enumerate(autotexts_int):
        text_color = "black" 
        at.set_color(text_color) 
        at.set_fontweight("normal")
        at.set_fontsize(global_font_size) # <-- Usa global_font_size
    
    # --- 6. Configurar Etiquetas Externas (SOLO Label) ---
    kw_ext = dict(arrowprops=dict(arrowstyle="-", color='black'), zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw_ext["arrowprops"].update({"connectionstyle": connectionstyle})
        
        label_text = f"{labels[i]}" 
        
        ax.annotate(label_text, xy=(x, y), xytext=(1.1*np.sign(x), 1.15*y),
                    horizontalalignment=horizontalalignment, fontsize=global_font_size, # <-- Usa global_font_size
                    color="black", **kw_ext)

    # --- 7. Limpieza y Anotación (Total ARRIBA y a la derecha) ---
    ax.set_title("") 
    ax.axis('equal')
    
    # [LÍNEA MODIFICADA]
    ax.text(1.02, 0.98, title_text, 
             transform=ax.transAxes, fontsize=int(global_font_size * 0.8), # <-- CAMBIO v24.3: Tamaño relativo
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', lw=0.5, alpha=0.8))
    
    # --- 8. Guardado (Si es standalone) ---
    if _ax_was_none and save_path_prefix:
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, bbox_inches='tight', dpi=300, format=fmt)
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"No se pudo guardar el gráfico en {fig_path}: {e}")
        
        plt.close(fig) 
            
    return ax


def plot_mapa_embudo_pie(total_universo, total_relevante, ax=None, save_path_prefix=None, **kwargs):
    """
    (v24.2) Gráfico de Torta 1: El Mapa del Embudo
    - CORRECCIÓN: Cálculo de 'total_filtrados' añadido.
    - Lógica de color: Menos % = Más Claro
    """
    # --- CÁLCULO CORREGIDO (Esta línea faltaba) ---
    total_filtrados = total_universo - total_relevante
    
    labels = ['Universo Relevante (A Clasificar)', 'Filtrados (No Relevantes)']
    # sizes = [~18.2% (Pequeña), ~81.8% (Grande)]
    sizes = [total_relevante, total_filtrados] 
    
    # --- LÓGICA DE COLOR SUAVE (Pequeña=Clara, Grande=Oscura) ---
    # Asignamos [Clara, Oscura-Media] para que coincida con [Pequeña, Grande]
    colors = [plt.cm.Greys(0.6), plt.cm.Greys(0.7)] 
    
    explode = [0.05, 0]
    title_text = f"Universo Total: {total_universo} SKUs"
    
    return _plot_pie_base(sizes, labels, colors, explode, title_text, ax, save_path_prefix, **kwargs)

def plot_zoom_clasificacion_pie(total_relevante, total_modelados_cv, total_predecibles_mase, total_no_predecibles_mase, 
                                ax=None, save_path_prefix=None, **kwargs):
    """
    (v24.2) Gráfico de Torta 2: El Zoom de Clasificación
    - Lógica de color CORREGIDA (Menos % = Más Claro) y tonos suaves.
    """
    total_no_modelados_cv = total_relevante - total_modelados_cv
    
    if total_modelados_cv != (total_predecibles_mase + total_no_predecibles_mase):
        logging.warning("Inconsistencia en los datos del Gráfico Zoom. La suma no coincide.")
        
    labels = [r'Predecibles (Núcleo, MASE < 1.0)', r'No Predecibles (MASE $\geq$ 1.0)', r'No Modelables (Filtro CV)']
    # sizes = [792 (37.4%), 252 (11.9%), 1071 (50.6%)] -> [Mediana, Pequeña, Grande]
    sizes = [total_predecibles_mase, total_no_predecibles_mase, total_no_modelados_cv] 
    
    # --- LÓGICA DE COLOR SUAVE (Pequeña=Clara, Mediana=Media, Grande=Oscura) ---
    # Asignamos [Media, Clara, Oscura-Media] para que coincida con [Mediana, Pequeña, Grande]
    colors = [plt.cm.Greys(0.7), plt.cm.Greys(0.6), plt.cm.Greys(0.8)]
    
    explode = [0, 0.05, 0]
    title_text = f"Universo Relevante: {total_relevante} SKUs"
    
    return _plot_pie_base(sizes, labels, colors, explode, title_text, ax, save_path_prefix, **kwargs)


def plot_ganadores_barplot(df_resumen_filtrado_dataset, 
                           ax=None, save_path_prefix=None, save_formats=['png', 'eps']):
    """
    (v24.6) Gráfico 3: Conteo de Modelos Ganadores (Nivel Tesis)
    - Corregido error 'get_bbox' (NoneType) iterando sobre ax.containers.
    - Se mantiene hue='Modelo' para compatibilidad con Seaborn >= 0.14.
    - Se calculan las etiquetas (N y %) dentro del bucle de etiquetado.
    - Se mantienen las mejoras estéticas (sin spines, título, etc.).
    """
    if ax is None:
        # Configuración estándar si el gráfico es individual
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": 14, "text.color": "black", "axes.labelcolor": "black",
            "xtick.color": "black", "ytick.color": "black"
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False

    if df_resumen_filtrado_dataset.empty or df_resumen_filtrado_dataset.iloc[:, 0].sum() == 0:
        ax.text(0.5, 0.5, "Sin Productos Predecibles", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title("Conteo de Modelos Ganadores (MASE < 1)")
        sns.despine(ax=ax, left=True, bottom=True); ax.set_xticks([]); ax.set_yticks([])
    else:
        df_plot = df_resumen_filtrado_dataset.reset_index().rename(columns={df_resumen_filtrado_dataset.columns[0]: 'Conteo', 'Modelo': 'Modelo'})
        df_plot = df_plot.sort_values(by='Conteo', ascending=False)
        
        # --- MODIFICACIÓN CLAVE (1): Se mantiene hue='Modelo' para Seaborn 0.14+ ---
        sns.barplot(x='Modelo', y='Conteo', data=df_plot, ax=ax, 
                    palette='Greys_r', hue='Modelo', legend=False) 
        
        ax.set_title("") # Título principal eliminado
        
        total_n = df_plot['Conteo'].sum()
        label_text = f"Núcleo Predecible (MASE < 1.0)\nN={total_n} SKUs"
        
        ax.text(0.98, 0.98, label_text,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', lw=0.5, alpha=0.8))

        # --- MODIFICACIÓN CLAVE (2): Iterar sobre los contenedores ---
        # Con hue='Modelo', ax.containers es una lista de contenedores (uno por barra)
        # Se debe etiquetar cada uno individualmente.
        if ax.containers:
            for container in ax.containers:
                # La etiqueta (N y %) se genera para cada barra
                labels = [f"{c.get_height()/total_n:.1%}\n(N={int(c.get_height())})" for c in container]
                ax.bar_label(container, labels=labels, fontsize=11, padding=3)
        
        ax.set_xlabel("Modelo Ganador", fontsize=14)
        ax.set_ylabel("Conteo de SKUs", fontsize=14)
        
        sns.despine(ax=ax) # Elimina bordes superior y derecho
        
        # Ajustar el límite Y para dar espacio a las etiquetas
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15) # Aumentado a 1.15 para más espacio

    if is_standalone and save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, format=fmt, dpi=300, bbox_inches='tight')
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"Error al guardar {fig_path}: {e}")
        plt.close(fig)
        
    return ax


def plot_mase_distribution(df_full_results, main_metric, threshold, 
                           ax=None, save_path_prefix=None, save_formats=['png', 'eps']):
    """
    (v24.6 - Modificación Tesis)
    - Título del gráfico eliminado (se usará el \caption de LaTeX).
    """
    if ax is None:
        # Configuración estándar si el gráfico es individual
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": 14, "text.color": "black", "axes.labelcolor": "black",
            "xtick.color": "black", "ytick.color": "black"
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False
    
    col_valor_ganador = f'{main_metric}_ganador'
    
    if col_valor_ganador not in df_full_results.columns:
        ax.text(0.5, 0.5, "Columna de Métrica no Encontrada", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title(f"Distribución de {main_metric} (Ganador)")
    else:
        mase_values = df_full_results[df_full_results[col_valor_ganador] < threshold][col_valor_ganador].dropna()
        if mase_values.empty:
            ax.text(0.5, 0.5, "Sin Productos Predecibles", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title(f"Distribución de {main_metric} (Ganador)");
            sns.despine(ax=ax, left=True, bottom=True); ax.set_xticks([]); ax.set_yticks([])
        else:
            sns.histplot(mase_values, ax=ax, kde=True, bins=20, color='#CCCCCC', edgecolor='black') # Color de barra más claro
            mean_val = mase_values.mean(); median_val = mase_values.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Media ({mean_val:.2f})', linewidth=2)
            ax.axvline(median_val, color='orange', linestyle='-', label=f'Mediana ({median_val:.2f})', linewidth=2)
            
            # --- MODIFICACIÓN CLAVE ---
            # El título se elimina, se deja para el caption de LaTeX.
            ax.set_title("", fontsize=16, pad=20) 
            
            ax.set_xlabel(f"Valor de {main_metric}", fontsize=14); ax.set_ylabel("Frecuencia (Productos)", fontsize=14)
            
            # --- MODIFICACIÓN ESTÉTICA ---
            # Se quita el borde de la leyenda para un look más limpio
            ax.legend(frameon=True, loc='upper right'); 
            
            sns.despine(ax=ax)

    if is_standalone and save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, format=fmt, dpi=300, bbox_inches='tight')
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"Error al guardar {fig_path}: {e}")
        plt.close(fig)
    
    return ax


def plot_sesgo_boxplot(df_full_results, main_metric, threshold, winning_col, 
                       ax=None, save_path_prefix=None, save_formats=['png', 'eps']):
    """
    (v24.7 - Modificación Tesis)
    - Añadido 'showfliers=False' para hacer zoom en el IQR y la mediana.
    - Título del gráfico eliminado.
    - Leyenda con borde reactivado (frameon=True) para consistencia.
    """
    if ax is None:
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": 14, "text.color": "black", "axes.labelcolor": "black",
            "xtick.color": "black", "ytick.color": "black"
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False
    
    col_valor_ganador = f'{main_metric}_ganador'
    
    if 'ME_ganador' not in df_full_results.columns or winning_col not in df_full_results.columns:
        ax.text(0.5, 0.5, "Columna 'ME_ganador' no Encontrada", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de Sesgo (ME) por Modelo")
    else:
        df_predecibles = df_full_results[(df_full_results[col_valor_ganador] < threshold)].dropna(subset=[winning_col, 'ME_ganador'])
        if df_predecibles.empty:
            ax.text(0.5, 0.5, "Sin Productos Predecibles", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de Sesgo (ME) por Modelo")
            sns.despine(ax=ax, left=True, bottom=True); ax.set_xticks([]); ax.set_yticks([])
        else:
            order = sorted(df_predecibles[winning_col].unique())
            
            sns.boxplot(x=winning_col, y='ME_ganador', data=df_predecibles, ax=ax, 
                        palette='Greys_r', order=order, hue=winning_col, legend=False,
                        showfliers=False) 
            
            ax.axhline(0, color='black', linestyle='--', label='Sesgo Cero (Ideal)')
            ax.set_title("", fontsize=16, pad=20) 
            ax.set_xlabel("Modelo Ganador", fontsize=14)
            ax.set_ylabel("Error Medio (ME)\n(Negativo=Sobreestima, Positivo=Subestima)", fontsize=14)
            
            # --- MODIFICACIÓN: Borde de leyenda reactivado ---
            ax.legend(loc='upper right', frameon=True, edgecolor='gray'); 
            sns.despine(ax=ax)

    if is_standalone and save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, format=fmt, dpi=300, bbox_inches='tight')
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"Error al guardar {fig_path}: {e}")
        plt.close(fig)
        
    return ax


def plot_risk_ratio_boxplot(df_full_results, main_metric, threshold, winning_col, 
                            ax=None, save_path_prefix=None, save_formats=['png', 'eps']):
    """
    (v24.8 - Modificación Tesis)
    - Añadido 'showfliers=False'.
    - Título eliminado.
    - Leyenda MOVIDA a 'upper left' para evitar solapamiento.
    - Borde de leyenda reactivado (frameon=True).
    - Límite Y superior AUMENTADO manualmente a 1.5 para dar "aire".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False
    
    col_valor_ganador = f'{main_metric}_ganador'
    
    if 'RMSE_ganador' not in df_full_results.columns or 'MAE_ganador' not in df_full_results.columns:
        ax.text(0.5, 0.5, "Columnas 'RMSE/MAE_ganador' no Encontradas", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de Ratio de Riesgo (RMSE/MAE)")
    else:
        df_predecibles = df_full_results.loc[
            (df_full_results[col_valor_ganador] < threshold)
        ].dropna(subset=[winning_col, 'RMSE_ganador', 'MAE_ganador']).copy()
        
        df_predecibles['Risk_Ratio'] = df_predecibles['RMSE_ganador'] / (df_predecibles['MAE_ganador'] + 1e-6)
        df_predecibles['Risk_Ratio'] = df_predecibles['Risk_Ratio'].replace([np.inf, -np.inf], np.nan)
        df_predecibles.dropna(subset=['Risk_Ratio'], inplace=True)
        
        if df_predecibles.empty:
            ax.text(0.5, 0.5, "Sin Productos Predecibles", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de Ratio de Riesgo (RMSE/MAE)")
            sns.despine(ax=ax, left=True, bottom=True); ax.set_xticks([]); ax.set_yticks([])
        else:
            order = sorted(df_predecibles[winning_col].unique())

            sns.boxplot(x=winning_col, y='Risk_Ratio', data=df_predecibles, ax=ax, 
                        palette='Greys_r', order=order, hue=winning_col, legend=False,
                        showfliers=False)
            
            ax.axhline(1, color='black', linestyle='--', label='Ratio Ideal (Errores Idénticos)')
            ax.set_title("", fontsize=16, pad=20)
            ax.set_xlabel("Modelo Ganador", fontsize=14)
            ax.set_ylabel("Ratio de Riesgo (RMSE/MAE)\n(> 1 = Errores grandes)", fontsize=14)
            
            # --- MODIFICACIÓN: Borde reactivado y ubicación cambiada ---
            ax.legend(loc='upper left', frameon=True, edgecolor='gray'); 
            
            # --- MODIFICACIÓN: Límites Y ajustados manualmente ---
            ax.set_ylim(bottom=0.9, top=1.5); # Ajustado de 0.9 a 1.5
            sns.despine(ax=ax)

    if is_standalone and save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, format=fmt, dpi=300, bbox_inches='tight')
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"Error al guardar {fig_path}: {e}")
        plt.close(fig)
        
    return ax


def plot_r2_distribution_boxplot(df_full_results, main_metric, threshold, winning_col, 
                                 ax=None, save_path_prefix=None, save_formats=['png', 'eps']):
    """
    (v24.8 - Modificación Tesis)
    - Añadido 'showfliers=False' para hacer zoom en el IQR y la mediana.
    - Título del gráfico eliminado.
    - Leyenda MOVIDA a 'upper left' (para evitar solapamiento).
    - Borde de leyenda reactivado (frameon=True).
    - Límite Y superior AUMENTADO manualmente a 0.5.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False
    
    col_valor_ganador = f'{main_metric}_ganador'
    
    if 'R2_ganador' not in df_full_results.columns:
        ax.text(0.5, 0.5, "Columna 'R2_ganador' no Encontrada", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de R² (Ruido de la Serie)")
    else:
        df_predecibles = df_full_results[(df_full_results[col_valor_ganador] < threshold)].dropna(subset=[winning_col, 'R2_ganador'])
        if df_predecibles.empty:
            ax.text(0.5, 0.5, "Sin Productos Predecibles", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray'); ax.set_title("Análisis de R² (Ruido de la Serie)")
            sns.despine(ax=ax, left=True, bottom=True); ax.set_xticks([]); ax.set_yticks([])
        else:
            order = sorted(df_predecibles[winning_col].unique())

            # --- MODIFICACIÓN CLAVE: Añadido showfliers=False ---
            sns.boxplot(x=winning_col, y='R2_ganador', data=df_predecibles, ax=ax, 
                        palette='Greys_r', order=order, hue=winning_col, legend=False,
                        showfliers=False) # <-- ESTA ES LA MODIFICACIÓN
            
            ax.axhline(0, color='black', linestyle='--', label='Benchmark (Modelo de Media)')
            
            # --- MODIFICACIÓN: Título eliminado ---
            ax.set_title("", fontsize=16, pad=20) 
            
            ax.set_xlabel("Modelo Ganador", fontsize=14)
            ax.set_ylabel("R² Out-of-Sample (CV)\n(> 0 = Supera la media)", fontsize=14)
            
            # --- MODIFICACIÓN: Borde reactivado y ubicación cambiada ---
            ax.legend(loc='upper left', frameon=True, edgecolor='gray'); 
            
            # --- MODIFICACIÓN: Límite Y superior ajustado para "aire" ---
            ax.set_ylim(top=0.5) 
            
            sns.despine(ax=ax)

    if is_standalone and save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        for fmt in save_formats:
            fig_path = f"{save_path_prefix}.{fmt}"
            try:
                fig.savefig(fig_path, format=fmt, dpi=300, bbox_inches='tight')
                logging.info(f"Gráfico individual guardado en: {fig_path}")
            except Exception as e:
                logging.error(f"Error al guardar {fig_path}: {e}")
        plt.close(fig)
        
    return ax


# =============================================================================
# --- SECCIÓN 6: ANÁLISIS DE CASOS DE ESTUDIO (NB 05) ---
# =============================================================================

def analizar_y_graficar_casos_de_estudio(codigos_productos, df_ventas_completo, df_resultados, figures_dir, 
                                        target_column='cantidad', 
                                        winning_col='modelo_ganador_MASE', 
                                        metric_col='MASE_ganador', 
                                        models_available=['Lineal', 'Cuadrático', 'KNN', 'SARIMA']):
    """
    (v24.9 - Modificación Tesis)
    - El eje X ahora muestra AÑOS en lugar de 'Mes (número)' para
      mejorar la interpretabilidad académica.
    """
    print(f"\n--- Analizando Casos de Estudio Específicos (Target: {target_column}) ---")
    
    if 'FECHA_MES_NRO' not in df_ventas_completo.columns and not df_ventas_completo.empty:
        df_ventas_completo = df_ventas_completo.sort_values(by='mes')
        min_year = df_ventas_completo['mes'].dt.year.min()
        df_ventas_completo['FECHA_MES_NRO'] = (df_ventas_completo['mes'].dt.year - min_year) * 12 + df_ventas_completo['mes'].dt.month
    elif df_ventas_completo.empty:
         print("Advertencia: df_ventas_completo está vacío.")
         return
    
    for codigo in codigos_productos:
        df_producto = df_ventas_completo[df_ventas_completo['codigo_producto'] == codigo].copy().sort_values('mes')
        if df_producto.empty:
            print(f" - ADVERTENCIA: No se encontraron datos de ventas para el producto {codigo}.")
            continue
            
        info_resultado = df_resultados[df_resultados['codigo_producto'] == str(codigo)]
        if info_resultado.empty:
            print(f" - ADVERTENCIA: No se encontraron resultados del modelo para el producto {codigo}.")
            continue

        mejor_modelo = info_resultado[winning_col].iloc[0] if winning_col in info_resultado.columns else 'N/A'
        valor_metrica = info_resultado[metric_col].iloc[0] if metric_col in info_resultado.columns else np.nan
        nombre_articulo = info_resultado['descripcion_producto'].iloc[0] if 'descripcion_producto' in info_resultado.columns else f'Producto {codigo}'

        if pd.isna(mejor_modelo) or mejor_modelo == 'N/A':
             print(f" - INFO: No hay modelo ganador para el producto {codigo}. No se graficará modelo.")
             mejor_modelo = 'N/A' 
        
        X = df_producto[['FECHA_MES_NRO']].values
        y = df_producto[target_column].values
        
        if len(X) < 2: 
             print(f" - INFO: Datos insuficientes para graficar modelo para {codigo}.")
             mejor_modelo = 'N/A'
        
        X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1) if len(X) >= 2 else X 

        # --- INICIO: NUEVA LÓGICA PARA ETIQUETAS DE EJE X (AÑOS) ---
        df_producto['anio'] = df_producto['mes'].dt.year
        # Agrupar por año y tomar el primer 'FECHA_MES_NRO' de ese año
        tick_data = df_producto.groupby('anio')['FECHA_MES_NRO'].min().reset_index()
        
        # Queremos ticks al inicio de cada año que esté presente
        xticks_loc = tick_data['FECHA_MES_NRO'].values
        xticks_lab = tick_data['anio'].values.astype(str)
        # --- FIN: NUEVA LÓGICA ---

        plt.style.use('default')
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(X, y, color="black", label='Datos reales (F_48_NP)', zorder=3, s=20)
        
        label_modelo = f'{mejor_modelo} (In-Sample Fit)' if mejor_modelo != 'N/A' else 'Modelo no disponible'
        y_plot = None
        
        if mejor_modelo != 'N/A' and len(X) >= 2:
            try:
                 with warnings.catch_warnings(): 
                    warnings.simplefilter("ignore")
                    if mejor_modelo == "Lineal" and "Lineal" in models_available:
                        modelo = LinearRegression().fit(X, y)
                        y_plot = modelo.predict(X_plot)
                    elif mejor_modelo == "Cuadrático" and "Cuadrático" in models_available:
                        poly = PolynomialFeatures(degree=2)
                        modelo = LinearRegression().fit(poly.fit_transform(X), y)
                        y_plot = modelo.predict(poly.transform(X_plot))
                    elif mejor_modelo == "KNN" and "KNN" in models_available:
                        k_knn_usado = info_resultado['K_KNN_USADO'].iloc[0]
                        k_knn_usado = int(k_knn_usado) if not pd.isna(k_knn_usado) else 3
                        n_neighbors = min(k_knn_usado, len(df_producto) -1)
                        n_neighbors = max(1, n_neighbors)
                        modelo = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
                        y_plot = modelo.predict(X_plot)
                    elif mejor_modelo == "SARIMA" and "SARIMA" in models_available and len(y) >= 6:
                        X_sarima = X.reshape(-1, 1)
                        auto_model = pm.auto_arima(y, X=X_sarima, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action='ignore', maxiter=50)
                        if auto_model:
                             y_pred_sarima_in_sample = auto_model.predict_in_sample(X=X_sarima)
                             if len(X) == len(y_pred_sarima_in_sample):
                                  smoother = KNeighborsRegressor(n_neighbors=min(5, len(X))).fit(X, y_pred_sarima_in_sample)
                                  y_plot = smoother.predict(X_plot)
                             else: 
                                  ax.scatter(X, y_pred_sarima_in_sample, color="#666666", marker='x', label=f'{mejor_modelo} (pred. puntual)', zorder=2, s=30)
                        else: label_modelo = f'{mejor_modelo} (fallido)'
            except Exception as e:
                 print(f" - ERROR al graficar modelo {mejor_modelo} para {codigo}: {e}")
                 label_modelo = f'{mejor_modelo} (error graf.)'

        if y_plot is not None: 
            ax.plot(X_plot, y_plot, color="#666666", label=label_modelo, zorder=2)
        
        # --- MODIFICADO: Aplicar etiquetas de eje X (Años) ---
        ax.set_xlabel("Año", fontsize=14, color='#222222') # Etiqueta actualizada
        
        if len(xticks_loc) > 0:
            ax.set_xticks(xticks_loc)
            ax.set_xticklabels(xticks_lab, rotation=0, ha='center')
        else:
            # Fallback por si acaso
            ax.set_xlabel("Período (Mes)", fontsize=14, color='#222222')
        # --- FIN DE LA MODIFICACIÓN ---
        
        ylabel = "Cantidad de producto vendido (unidades)" if target_column == 'cantidad' else "Monto de venta (Gs.)"
        ax.set_ylabel(ylabel, fontsize=14, color='#222222')
        ax.grid(False)
        sns.despine(ax=ax)

        max_len = 65
        titulo_grafico = (nombre_articulo[:max_len] + '...') if len(nombre_articulo) > max_len else nombre_articulo
        
        # --- MODIFICADO: Título del gráfico eliminado para Tesis ---
        # (El título estará en el \caption de LaTeX)
        ax.set_title("", loc='center', y=1.0) # Título eliminado para publicación
        
        if not pd.isna(valor_metrica):
             metric_name = metric_col.split('_')[0]
             text_color = 'black'
             metric_text_val = f'${metric_name}_{{CV}} = {valor_metrica:.3f}$' 

             if metric_name == "MASE":
                 if valor_metrica < 1.0:
                     metric_text_val = f'$\\mathbf{{{metric_name}_{{CV}} = {valor_metrica:.3f}}}$'
                 elif valor_metrica >= 1.0:
                     text_color = 'red'
                     metric_text_val = f'${metric_name}_{{CV}} = {valor_metrica:.3f}$'

             if mejor_modelo != 'N/A':
                 metric_text_full = f'$\\mathbf{{{mejor_modelo}}}$\n{metric_text_val}'
             else:
                 metric_text_full = metric_text_val

             ax.text(0.03, 0.97, metric_text_full, 
                     transform=ax.transAxes, 
                     fontsize=11, 
                     verticalalignment="top", 
                     color=text_color,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
        
        # --- MODIFICADO: Leyenda con borde (cuadradito) ---
        ax.legend(loc='upper right', frameon=True, edgecolor='gray', fontsize=10)
        
        nombre_limpio = "".join(c for c in str(nombre_articulo) if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")[:30]
        os.makedirs(figures_dir, exist_ok=True) 
        
        if pd.isna(valor_metrica):
            metric_name = metric_col.split('_')[0]
            
        file_name_prefix = f"{target_column}_{metric_name}_{valor_metrica:.2f}_{mejor_modelo}_{codigo}_{nombre_limpio}"
        base_path = os.path.join(figures_dir, file_name_prefix)
        
        print(f" - Guardando gráficos para '{nombre_articulo}' ({codigo}):")
        
        fig_path_png = f"{base_path}.png"
        try:
            plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            print(f"   - {fig_path_png}")
        except Exception as e:
            print(f"   - Error al guardar PNG: {e}")
        
        fig_path_eps = f"{base_path}.eps"
        try:
            plt.savefig(fig_path_eps, format='eps', bbox_inches='tight') 
            print(f"   - {fig_path_eps}")
        except Exception as e:
            print(f"   - Error al guardar EPS: {e}")
        
        plt.close(fig)