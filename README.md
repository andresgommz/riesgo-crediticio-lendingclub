# Riesgo de incumplimiento crediticio (LendingClub)

Proyecto integrador de Inteligencia Artificial, Ingeniería de Sistemas, UNAB.
Integrantes: Andrés Gómez y Rodolfo Hernández.
Profesor: Ph.D. Andrés Felipe Jerez Ariza.

## Enlaces

- Aplicación en Streamlit: https://riesgo-crediticio-lendingclub.streamlit.app
- Notebook en Google Colab: https://drive.google.com/file/d/1cQPE1zeIqeSKwu6wxwsXp9pMm34SCheg/view?usp=sharing
- Repositorio: https://github.com/andresgommz/riesgo-crediticio-lendingclub

## Problema

Con la información disponible al originar un préstamo peer-to-peer de LendingClub, ¿es posible anticipar si terminará pagado (Fully Paid) o incumplido (Charged Off)?

Es aprendizaje supervisado de clasificación binaria, con una muestra reproducible de 30,000 préstamos (random_state=42). De ella quedan 17,264 préstamos con desenlace definitivo (20.1% de default).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Project.ipynb` | Notebook de Colab con las dos fases: propuesta (1er semestre) e implementación (2do semestre). |
| `app.py` | Frontend de la aplicación en Streamlit. |
| `model.py` | Backend: carga el modelo y aplica el mismo preprocesamiento del notebook. |
| `modelo_riesgo.joblib` | Modelo final (Logistic Regression + SMOTE) y objetos de preprocesamiento. |
| `prestamos_prueba.csv` | 3,453 préstamos del conjunto de prueba, usados en la demostración. |
| `requirements.txt` | Librerías necesarias para la aplicación. |

## Resultados (conjunto de prueba)

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (seleccionado) | 0.648 | 0.307 | 0.593 | **0.404** | **0.682** |
| Random Forest | 0.606 | 0.289 | 0.655 | 0.401 | 0.680 |
| XGBoost | 0.636 | 0.296 | 0.587 | 0.393 | 0.676 |

El modelo detecta el 59% de los préstamos que terminan en default. Entre los préstamos que considera seguros, la tasa de default baja de 20.1% a 13.4%.

## Ejecutar localmente

Requiere **Python 3.13** (probado con 3.13.7, 3.13.17 y 3.13.18). Con Python 3.14 la instalación falla, porque scikit-learn 1.6.1 y pandas 2.2.3 no tienen versiones para esa versión.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

La aplicación abre en http://localhost:8501.

## Nota sobre versiones

`requirements.txt` fija las mismas versiones que imprime la última celda del notebook en Colab: scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6 y joblib 1.6.0. Se verificó que con estas versiones `modelo_riesgo.joblib` carga sin advertencias y reproduce en los 3,453 préstamos de prueba la misma matriz de confusión y el mismo AUC (0.6816) del notebook. Si se vuelve a ejecutar el notebook en Colab, hay que actualizar `requirements.txt` con las versiones que imprima esa celda.
