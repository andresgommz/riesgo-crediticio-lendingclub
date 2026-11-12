import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model

st.title("Riesgo de incumplimiento crediticio")
st.markdown("Predicción de **default** (Charged Off) de un préstamo de LendingClub a partir de la información "
            "disponible al momento de originarlo. Modelo: **" + model.artefactos['nombre_modelo'] + "**.")

prueba = pd.read_csv("prestamos_prueba.csv")
tasa_base = model.artefactos['tasa_default']

st.header("1. Datos del préstamo")
modo = st.selectbox("¿Cómo quiere ingresar el préstamo?",
                    ["Seleccionar un préstamo real del conjunto de prueba", "Ingresar datos manualmente"])

if modo == "Seleccionar un préstamo real del conjunto de prueba":
    indice = st.slider("Préstamo número", 0, len(prueba) - 1, 0)
    fila = prueba.iloc[indice]
    st.dataframe(prueba.iloc[[indice]].drop(columns=['default']))
    datos = fila.drop(labels=['default']).to_dict()
    real = int(fila['default'])
else:
    real = None
    sub_grades = [g + str(n) for g in 'ABCDEFG' for n in range(1, 6)]
    tasas = model.artefactos['int_rate_por_sub_grade']

    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("Monto solicitado (loan_amnt, USD)", min_value=1000.0, max_value=40000.0, value=12000.0, step=500.0)
        term = st.selectbox("Plazo (term, meses)", [36, 60])
        sub_grade = st.selectbox("Subcalificación (sub_grade)", sub_grades, index=7)
        int_rate = st.number_input("Tasa de interés (int_rate, %)", min_value=5.0, max_value=31.0, value=float(round(tasas.get(sub_grade, 13.0), 2)))
        annual_inc = st.number_input("Ingreso anual (annual_inc, USD)", min_value=0.0, max_value=1000000.0, value=65000.0, step=1000.0)
        emp_length = st.selectbox("Antigüedad laboral (emp_length)", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], index=10)
        # OTHER se excluye: solo tiene 7 casos en los datos y su coeficiente escalado no es confiable
        home_ownership = st.selectbox("Vivienda (home_ownership)", ['MORTGAGE', 'RENT', 'OWN'])
        verification_status = st.selectbox("Verificación de ingreso (verification_status)", ['Not Verified', 'Source Verified', 'Verified'])
    with col2:
        purpose = st.selectbox("Propósito (purpose)", model.artefactos['purpose_opciones'])
        application_type = st.selectbox("Tipo de solicitud (application_type)", ['Individual', 'Joint App'])
        dti = st.slider("Relación deuda/ingreso (dti, %)", 0.0, 60.0, 17.7)
        revol_util = st.slider("Uso de crédito rotativo (revol_util, %)", 0.0, 100.0, 52.7)
        revol_bal = st.number_input("Saldo rotativo (revol_bal, USD)", min_value=0.0, max_value=500000.0, value=11000.0, step=500.0)
        open_acc = st.number_input("Cuentas abiertas (open_acc)", min_value=0, max_value=80, value=11)
        total_acc = st.number_input("Cuentas totales (total_acc)", min_value=1, max_value=150, value=23)
        mort_acc = st.number_input("Hipotecas (mort_acc)", min_value=0, max_value=30, value=1)

    datos = {'loan_amnt': loan_amnt, 'term': term, 'sub_grade': sub_grade, 'int_rate': int_rate, 'annual_inc': annual_inc, 'emp_length': emp_length, 'home_ownership': home_ownership, 'verification_status': verification_status, 'purpose': purpose, 'application_type': application_type, 'dti': dti, 'revol_util': revol_util, 'revol_bal': revol_bal, 'open_acc': open_acc, 'total_acc': total_acc, 'mort_acc': mort_acc}

st.header("2. Ejecutar el modelo")
run = st.button("Run")

if run:
    probabilidad, clase = model.predecir(datos)

    st.header("3. Resultado")
    if clase == 1:
        st.error("Riesgo ALTO: el modelo predice que el préstamo terminará en default (Charged Off).")
    else:
        st.success("Riesgo BAJO: el modelo predice que el préstamo será pagado (Fully Paid).")

    st.write("Probabilidad estimada de default: **" + str(round(probabilidad * 100, 1)) + "%** "
             "(umbral de decisión: " + str(round(model.umbral * 100)) + "%; tasa promedio de default en los datos: "
             + str(round(tasa_base * 100, 1)) + "%).")

    if real is not None:
        resultado_real = "Charged Off (default)" if real == 1 else "Fully Paid"
        if real == clase:
            st.info("Resultado real de este préstamo: " + resultado_real + ". El modelo acertó.")
        else:
            st.warning("Resultado real de este préstamo: " + resultado_real + ". El modelo se equivocó.")

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.barh(['Este préstamo'], [probabilidad], color='#c0392b' if clase == 1 else '#27ae60')
    ax.axvline(model.umbral, color='black', linestyle='--', label='Umbral de decisión')
    ax.axvline(tasa_base, color='gray', linestyle=':', label='Tasa promedio de default')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilidad de default')
    ax.legend(loc='lower right', fontsize=8)
    st.pyplot(fig)

st.header("Desempeño del modelo en el conjunto de prueba")
st.table(pd.DataFrame(model.artefactos['metricas_test'], index=['Valor']).T.round(3))

col1, col2 = st.columns(2)

matriz = np.array(model.artefactos['matriz_confusion'])
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(matriz, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, matriz[i, j], ha='center', va='center', fontsize=12,
                color='white' if matriz[i, j] > matriz.max() / 2 else 'black')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fully Paid', 'Charged Off'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Fully Paid', 'Charged Off'])
ax.set_xlabel('Predicho')
ax.set_ylabel('Real')
ax.set_title('Matriz de confusión (test)')
fig.tight_layout()
with col1:
    st.pyplot(fig)

importancias = pd.Series(model.artefactos['importancias'])
if model.artefactos['nombre_modelo'] == 'Logistic Regression':
    importancias = importancias.reindex(importancias.abs().sort_values().index).tail(10)
    colores = ['#c0392b' if v > 0 else '#27ae60' for v in importancias]
    titulo = 'Coeficientes (rojo: más riesgo, verde: menos)'
else:
    importancias = importancias.sort_values().tail(10)
    colores = '#2980b9'
    titulo = 'Variables más importantes del modelo'
fig, ax = plt.subplots(figsize=(4, 3.5))
importancias.plot(kind='barh', ax=ax, color=colores)
ax.set_title(titulo, fontsize=10)
ax.tick_params(labelsize=7)
fig.tight_layout()
with col2:
    st.pyplot(fig)
