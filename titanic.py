import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Titanic: Predykcja Przeżycia", page_icon="🚢")


@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as file:
        return pickle.load(file)


model = load_model()

st.image(
    'https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG')

st.title("Przeżycie na Titanicu 🚢")
st.write(
    "Wprowadź dane pasażera korzystając z poniższych widgetów, aby sprawdzić, czy przetrwałby katastrofę, oraz poznać prawdopodobieństwo tego zdarzenia.")

st.header("Dane pasażera")

col1, col2 = st.columns(2)

with col1:
    pclass = st.radio("Klasa biletu", options=[1, 2, 3], index=2, help="1 = Pierwsza, 2 = Druga, 3 = Trzecia")
    sex = st.selectbox("Płeć", options=["Mężczyzna", "Kobieta"])
    age = st.slider("Wiek", min_value=0.0, max_value=100.0, value=30.0, step=0.5)

with col2:
    sibsp = st.number_input("Liczba rodzeństwa/małżonków na pokładzie", min_value=0, max_value=10, value=0)
    parch = st.number_input("Liczba rodziców/dzieci na pokładzie", min_value=0, max_value=10, value=0)
    fare = st.slider("Opłata za bilet", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

if st.button("Sprawdź szanse na przeżycie", type="primary"):

    sex_encoded = 1 if sex == "Kobieta" else 0

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    survival_prob = probabilities[1] * 100
    death_prob = probabilities[0] * 100

    st.divider()
    st.subheader("Wynik predykcji:")

    if prediction == 1:
        st.success("Model przewiduje, że ten pasażer **przeżyłby** katastrofę!")
        st.metric(label="Prawdopodobieństwo przeżycia", value=f"{survival_prob:.2f}%")
    else:
        st.error("Model przewiduje, że ten pasażer **nie przeżyłby** katastrofy.")
        st.metric(label="Prawdopodobieństwo śmierci", value=f"{death_prob:.2f}%")