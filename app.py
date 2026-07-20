import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Irrigation Ouarzazate", layout="wide")

# Charger les données et le modèle
df = pd.read_csv('donnees_meteo_ouarzazate_propre.csv')
modele = joblib.load('modele_prediction.pkl')

st.title("🌊 Dashboard - Prédiction des besoins en eau d'irrigation à Ouarzazate")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "🔍 Exploration", "🤖 Modèle ML", "🎛️ Simulateur"])

# ---------- ONGLET 1 : VUE D'ENSEMBLE ----------
with tab1:
    st.header("Vue d'ensemble du projet")
    st.write("""
    Ce dashboard analyse les données climatiques de la région de Ouarzazate (2005-2025) 
    pour estimer les besoins en eau d'irrigation à partir de la température et des précipitations.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Température moyenne", f"{df['T2M'].mean():.1f} °C")
    col2.metric("Précipitation moyenne", f"{df['PRECTOTCORR'].mean():.2f} mm/jour")
    col3.metric("Période couverte", "2005 - 2025")
    
    st.info("Source des données : NASA POWER (données climatiques ouvertes)")

# ---------- ONGLET 2 : EXPLORATION ----------
with tab2:
    st.header("Exploration des données climatiques")
    
    mois_noms = ['Jan','Fév','Mar','Avr','Mai','Juin','Juil','Août','Sep','Oct','Nov','Déc']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Température moyenne par mois")
        temp_par_mois = df.groupby('MOIS')['T2M'].mean()
        fig1, ax1 = plt.subplots()
        ax1.bar(mois_noms, temp_par_mois.values, color='orange')
        ax1.set_ylabel("Température (°C)")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Précipitations moyennes par mois")
        pluie_par_mois = df.groupby('MOIS')['PRECTOTCORR'].mean()
        fig2, ax2 = plt.subplots()
        ax2.bar(mois_noms, pluie_par_mois.values, color='steelblue')
        ax2.set_ylabel("Précipitations (mm)")
        st.pyplot(fig2)
    
    st.subheader("Évolution de la température moyenne annuelle")
    temp_par_annee = df.groupby('YEAR')['T2M'].mean()
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(temp_par_annee.index, temp_par_annee.values, marker='o', color='red')
    ax3.set_xlabel("Année")
    ax3.set_ylabel("Température moyenne (°C)")
    st.pyplot(fig3)

# ---------- ONGLET 3 : MODÈLE ML ----------
with tab3:
    st.header("Performance du modèle de Machine Learning")
    
    st.write("""
    Un modèle de **régression linéaire** a été entraîné pour estimer les précipitations 
    à partir de la température.
    """)
    
    col1, col2 = st.columns(2)
    col1.metric("Erreur moyenne (MAE)", "0.86 mm")
    col2.metric("Erreur quadratique (RMSE)", "2.05 mm")
    
    st.write("""
    **Observation :** un modèle plus complexe (Random Forest) a aussi été testé, 
    mais n'a pas amélioré les résultats (0.96 mm d'erreur), ce qui montre que la relation 
    entre température et précipitation est ici relativement simple.
    """)

# ---------- ONGLET 4 : SIMULATEUR ----------
with tab4:
    st.header("🎛️ Simulateur de prédiction")
    st.write("Choisis une température pour estimer la précipitation prédite par le modèle :")
    
    temperature_input = st.slider("Température (°C)", 
                                    min_value=float(df['T2M'].min()), 
                                    max_value=float(df['T2M'].max()), 
                                    value=20.0)
    
    prediction = modele.predict([[temperature_input]])[0]
    
    st.metric("Précipitation prédite", f"{prediction:.2f} mm")
    
    if temperature_input > 25:
        st.warning("⚠️ Température élevée : risque de stress hydrique, précipitations attendues faibles.")
    else:
        st.success("✅ Conditions climatiques modérées.")
