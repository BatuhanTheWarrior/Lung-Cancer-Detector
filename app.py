import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("lung_model.pkl", "rb"))

lang = st.selectbox("Language / Dil", ["English", "Türkçe"])

def t(en, tr):
    return en if lang == "English" else tr

st.title(t("Cancer Risk Predictor", "Kanser Risk Tahmin Aracı"))

age = st.slider(t("Age", "Yaş"), 18, 100)

smoking = st.selectbox(t("Do you smoke?", "Sigara kullanıyor musunuz?"), [0,1])
alcohol = st.selectbox(t("Alcohol consumption?", "Alkol tüketiyor musunuz?"), [0,1])
chronic = st.selectbox(t("Chronic disease?", "Kronik hastalığınız var mı?"), [0,1])
fatigue = st.selectbox(t("Fatigue?", "Yorgunluk hissediyor musunuz?"), [0,1])

if st.button(t("Predict", "Tahmin Et")):
    input_data = np.array([[age, smoking, alcohol, chronic, fatigue]])
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        risk = t("Low Risk", "Düşük Risk")
    elif prob < 0.7:
        risk = t("Medium Risk", "Orta Risk")
    else:
        risk = t("High Risk", "Yüksek Risk")

    st.subheader(risk)
    st.write(f"{t('Probability:', 'Olasılık:')} {prob:.2f}")

    st.warning(t(
        "This is NOT a medical diagnosis.",
        "Bu bir tıbbi teşhis değildir."
    ))
