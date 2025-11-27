# Metodología de Predicción de Demanda para Inventarios en el Sector de Distribución Ferretero

Este repositorio contiene el código fuente, los cuadernos de experimentación (*notebooks*) y los recursos computacionales desarrollados para el Trabajo Final de Grado de Ingeniería Industrial: **"Diseño de un Marco de Decisión Logística mediante Pronóstico Híbrido y Segmentación de Inventario: Caso de Estudio en el Sector de Distribución Ferretero"**.

## 📋 Descripción del Proyecto

El objetivo de este proyecto es desarrollar un marco de modelado híbrido que compare enfoques deterministas (Regresión), estocásticos (SARIMA) y de aprendizaje automático no paramétrico (KNN) para segmentar y predecir la demanda de un inventario mayorista de +11,000 SKUs.

El flujo de trabajo sigue una adaptación académica de la metodología **CRISP-DM**.

## 🚀 Estructura del Pipeline

El procesamiento se divide en 6 etapas secuenciales, documentadas en la carpeta `notebooks/`:

1.  **[01_Preprocesamiento](notebooks/01_Preprocesamiento.ipynb)**: Ingesta de datos crudos (.DBF), limpieza ETL y consolidación mensual.
2.  **[02_Analisis_y_Filtrado](notebooks/02_Analisis_y_Filtrado.ipynb)**: Aplicación del "Embudo de Selección". Filtros de 48 meses, detección de pandemia y outliers. Definición del Universo Relevante.
3.  **[03_Modelado](notebooks/03_Modelado.ipynb)**: Entrenamiento y validación cruzada (*Time Series Split*) de los modelos competidores. Selección de hiperparámetros.
4.  **[04_Analisis_de_Resultados](notebooks/04_Analisis_de_Resultados.ipynb)**: Evaluación estadística basada en MASE. Clasificación del inventario en Predecible vs. No Predecible.
5.  **[05_Analisis_Casos_de_Estudio](notebooks/05_Analisis_Casos_de_Estudio.ipynb)**: Auditoría visual de los modelos ganadores (Lineal, SARIMA, KNN) para validar coherencia logística.
6.  **[06_Entregable](notebooks/06_Entregable.ipynb)**: Generación de la "Maestra de Productos Predecibles" y exportación de resultados para la toma de decisiones.

## 🛠️ Requisitos de Instalación

El proyecto utiliza Python 3.12.10. Las dependencias principales son:
* `pandas` & `numpy`: Manipulación de datos.
* `scikit-learn`: Modelos de regresión, KNN y métricas.
* `pmdarima`: Implementación de Auto-ARIMA/SARIMA.
* `matplotlib` & `seaborn`: Visualización de datos.

Para replicar el entorno:
```bash

pip install -r requirements.txt

