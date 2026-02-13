# Marco de Decisión Logística: Pronóstico Híbrido y Segmentación de Inventario

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Finalizado-success)

## Resumen Ejecutivo

Este proyecto, desarrollado como Trabajo Final de Grado en Ingeniería Industrial, implementa un marco de modelado híbrido para la **predicción de demanda** y **segmentación de inventario** en el sector de distribución ferretero mayorista.

El sistema procesa un universo de más de **11,000 SKUs**, comparando enfoques deterministas, estocásticos y de aprendizaje automático para optimizar la toma de decisiones logísticas.

**Objetivos Técnicos:**
* Comparación de rendimiento entre modelos de Regresión Lineal, SARIMA (estocástico) y K-Nearest Neighbors (no paramétrico).
* Implementación de metodología CRISP-DM adaptada a la cadena de suministro.
* Validación estadística mediante métrica MASE (Mean Absolute Scaled Error).

## Stack Tecnológico

* **Lenguaje:** Python 3.12.10
* **Procesamiento de Datos:** Pandas, NumPy
* **Modelado y Machine Learning:** Scikit-learn, Pmdarima (Auto-ARIMA)
* **Visualización:** Matplotlib, Seaborn

## Arquitectura del Pipeline

El flujo de trabajo se estructura en seis etapas secuenciales diseñadas para garantizar la integridad de los datos y la robustez de las predicciones.

### 1. Ingesta y Preprocesamiento (ETL)
Transformación de datos crudos (`.DBF`) y consolidación temporal. Se asegura la calidad del dato antes de iniciar el análisis.
* *Notebook:* `01_Preprocesamiento.ipynb`

### 2. Análisis Exploratorio y Filtrado (EDA)
Aplicación de un "Embudo de Selección" para definir el universo relevante.
* Filtrado de histórico (ventanas de 48 meses).
* Detección de anomalías por impacto de pandemia y outliers estadísticos.
* *Notebook:* `02_Analisis_y_Filtrado.ipynb`

### 3. Entrenamiento y Validación (Modelado)
Entrenamiento de modelos competidores utilizando validación cruzada para series temporales (*Time Series Split*).
* Ajuste de hiperparámetros para Regresión, SARIMA y KNN.
* *Notebook:* `03_Modelado.ipynb`

### 4. Evaluación de Desempeño
Clasificación del inventario basada en la previsibilidad. Se utiliza el MASE como métrica principal para determinar la viabilidad de la automatización del pronóstico frente a métodos ingenuos (Naïve).
* *Notebook:* `04_Analisis_de_Resultados.ipynb`

### 5. Auditoría de Casos de Estudio
Validación visual y lógica de los modelos ganadores para asegurar la coherencia con la operativa logística real.
* *Notebook:* `05_Analisis_Casos_de_Estudio.ipynb`

### 6. Despliegue de Resultados
Generación de la "Maestra de Productos Predecibles" y exportación de datos finales para integración en sistemas ERP o dashboards de BI.
* *Notebook:* `06_Entregable.ipynb`

## Instalación y Reproducibilidad

Se recomienda ejecutar el proyecto en un entorno virtual para gestionar las dependencias correctamente.

```bash
# Clonar el repositorio
git clone <url-del-repositorio>

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
