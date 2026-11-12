import joblib
import numpy as np
import pandas as pd

artefactos = joblib.load('modelo_riesgo.joblib')

modelo = artefactos['modelo']
scaler = artefactos['scaler']
pca = artefactos['pca']
medianas = artefactos['medianas']
columnas = artefactos['columnas']
purpose_validos = artefactos['purpose_validos']
umbral = artefactos['umbral']

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
           '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}


def preparar(datos):
    fila = pd.DataFrame(0.0, index=[0], columns=columnas)

    annual_inc = datos['annual_inc'] if datos['annual_inc'] > 0 else np.nan

    fila['loan_amnt'] = datos['loan_amnt']
    fila['int_rate'] = datos['int_rate']
    fila['dti'] = datos['dti']
    fila['open_acc'] = datos['open_acc']
    fila['revol_util'] = min(datos['revol_util'], 100)
    fila['total_acc'] = datos['total_acc']
    fila['mort_acc'] = datos['mort_acc']
    fila['term'] = datos['term']
    fila['sub_grade'] = grade_map[datos['sub_grade'][0]] * 5 - 5 + int(datos['sub_grade'][1])
    fila['emp_length'] = emp_map.get(datos['emp_length'], np.nan)
    fila['annual_inc_log'] = np.log1p(annual_inc)
    fila['revol_bal_log'] = np.log1p(datos['revol_bal'])
    fila['loan_to_income_log'] = np.log1p(datos['loan_amnt'] / annual_inc)

    home = datos['home_ownership']
    if home in ['NONE', 'ANY']:
        home = 'OTHER'
    purpose = datos['purpose'] if datos['purpose'] in purpose_validos else 'other'

    for col in ['home_ownership_' + home,
                'verification_status_' + datos['verification_status'],
                'purpose_' + purpose,
                'application_type_' + datos['application_type']]:
        if col in columnas:
            fila[col] = 1

    fila = fila.fillna(medianas)
    return fila


def predecir(datos):
    fila = preparar(datos)
    X = scaler.transform(fila)
    if pca is not None:
        X = pca.transform(X)
    probabilidad = modelo.predict_proba(X)[0, 1]
    clase = int(probabilidad >= umbral)
    return probabilidad, clase
