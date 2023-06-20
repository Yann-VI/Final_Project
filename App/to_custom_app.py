import streamlit as st
import boto3
import os
import requests
import botocore
from io import BytesIO
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load


def main():
    st.title("Image Loader")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the image here (e.g., save it to a specific location)
        #image = uploaded_file.getbuffer()
        file_path = save_uploaded_file_to_temp(uploaded_file)
        save_image_to_s3(file_path, 'temp/test.jpg')
        st.success("Image loaded successfully!")
        st.image(uploaded_file)

        if st.button("Prediction"):
            # Send request to FastAPI server
            file_name = "uploaded_image.png"
            
            
            
            api_url = "https://wakeup-api.herokuapp.com/predict"  # Replace with your FastAPI server URL
            data = {"file_name": file_name}
            response = requests.post(api_url, json=data)

            if response.status_code == 200:
                result = response.json()
                st.write("Prediction Result:", result)
            else:
                st.write("Prediction Failed!")



with st.sidebar:
    choose = option_menu("Projet EnerPy", ["Contexte", "Data", "Exploration", "Modélisation", "Résultats","Conclusion"],
                         icons=['house', 'files', 'binoculars', 'gear', 'bar-chart-line','signpost'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Contexte":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contexte</p>', unsafe_allow_html=True)    
    
    st.image('thuc_contexte.jpg')
    
elif choose == "Data":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Données disponibles</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('thuc_data_01.jpg')
    with col2:
        st.image('thuc_data_02.jpg')
    with col3:
        st.image('thuc_data_03.jpg')
   
    tab1, tab2, tab3 = st.tabs(["Fiabilité", "Déséquilibré", "Volumineux"])
 
    with tab1:  
        st.markdown(, unsafe_allow_html=True)
 
    with tab2:
        st.image('thuc_exploration_01.jpg')

        
    with tab3:
        st.write('Le jeu de données comporte un total de 1 980 288 lignes et 32 colonnes.')
    
elif choose == "Exploration":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Exploration - DataViz</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('thuc_exploration_01.jpg')
    with col2:
        st.image('thuc_exploration_02.jpg')
    with col3:
        st.image('thuc_exploration_03.jpg')
    
    
    tab1, tab2, tab3, tab4 = st.tabs(["Phase 1", "Phase 2", "Phase 3","Phase 4"])
 
    with tab1:
        st.image('thuc_phase1_01.png')
       
    with tab2:
        st.image('thuc_phase2_01.png')

    with tab3:
        st.image('thuc_phase3_01.png')

    with tab4:
        st.image('thuc_phase4_01.png')
        liste_energies = ['Nucléaire (MW)','Thermique (MW)','Eolien (MW)','Solaire (MW)','Hydraulique (MW)','Bioénergies (MW)']
        s=df.groupby('Région')[liste_energies].sum().reset_index()
        st.bar_chart(data=s, x = 'Région', height=400, width=400, use_container_width=False)

elif choose == "Modélisation":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Machine Learning</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Modèles", "Variables explicatives", "Split train test", "Mise à l'échelle"])
 
    with tab1:
        st.markdown("""
            S'agissant de prédire la variable **Consommation** qui est une variable numérique quantitative, nous avons testé plusieurs modèles de régression, dont 4 principaux :
            - **Régression linéaire (Linear Regression)**
            - K plus proches voisins (K Neighbors Regressor)
            - Foret aléatoire (Random Forest Regressor)
            - Arbre de décisions (Decision Tree Regressor)
            et 3 autres qui ne sont pas présentés ici (calculs longs, mauvais scores, pas maitrisés à notre niveau)
            - Régression Vectorielle du Support (SVR)
            - Régression Polynomiale
            - LSTM (modèle basé sur les réseaux de neurones)
            """, unsafe_allow_html=True)  
    
    with tab2: 
#        st.subheader('Variables explicatives')
        st.markdown("""
            Notre variable cible : _Consommation (MW)_<br>
            Quelles variables de notre jeu de données ont du sens pour \"expliquer\" la consommation ?
            - sachant que l’électricité en tant que telle ne peut pas être stockée, en tout cas pas avec les technologies actuelles.
            - que la production "suit" la consommation (offre suit la demande)
            - que les données disponibles sont chronologiques
            """, unsafe_allow_html=True)
    
#    col1, col2 = st.columns( [0.5, 0.5])
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<p style="text-align: left;">Base</p>',unsafe_allow_html=True)
            st.markdown("""
            _Consommation (MW)_<br>
            Date<br>
            Heure
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<p style="text-align: left;">Essai 1</p>',unsafe_allow_html=True)
            st.markdown("""
            _Consommation (MW)_<br>
            Saison<br>
            Semaine<br>
            Nuit<br>
            Heure_sin<br>
            Heure_cos<br>
            """, unsafe_allow_html=True)

        
        with col3:
            st.markdown('<p style="text-align: left;">Essai 2</p>',unsafe_allow_html=True)
            st.markdown("""
            _Consommation (MW)_<br>
            Saison<br>
            Nuit<br>
            Semaine<br>
            Heure_sin<br>
            Heure_cos<br>
            Production(s)(MW)
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown('<p style="text-align: left;">Spécificité</p>',unsafe_allow_html=True)
            st.markdown("""
            numérique quantitative<br>
            ordinale hiérarchique<br>
            ordinale hierarchique<br>
            ordinale hiérarchique<br>
            variable circulaire<br>
            variable circulaire<br>
            numérique quantitative
            """, unsafe_allow_html=True)            

    
    with tab3:
#       st.subheader('Données d\'entrainement et données de test')
        st.markdown("""
            S'agissant de données chronologique, une attention particulières a été donnée à : 
            - consever l'ordre des échantillons
            - s'assurer que les données de test sont biens postérieures au données d'entrainement
            4 manières ont été expérimentées :
            - train_test_split 'classique'
            - **train_test_split avec shuffle=false et proportion 80/20**
            - TimeSeriesSplit
            - séparation manuelle (ex : train = années [2019, 2020], test = année[2021]<br>

            Le jeu de données retenu comprend 52560 échantillons, représentant les données consolidées au niveau de la France pour les années 2019, 2020 et 2021<br>
            RQ: excepté pour train_test_split, les 3 autres ont donné des résultats similaires.<br>
            RQ: plusieurs combinaisons de proportions ont été testées (80/20, 75/25), sans différence notable
            """, unsafe_allow_html=True)  
    
    with tab4:
#        st.subheader('Mise à l\'echelle et encodage')
        st.markdown("""
            - Concernant les variables 'Saison', 'Semaine', 'Nuit': 
                - déjà en numériques, ordonnées et hierarchiques
            - Concernant les variables 'Heure_sin' et 'Heure_cos'
                - déjà en numériques et au bon format
            - Concernant les variables 'Production(s)' , 4 mises à l'échelle on été testées :
                - rien
                - **StandartScaler**
                - MinMaxScaler
                - RobustScaler
            RQ: aucune différence notable concernant les différents scaler.
            """, unsafe_allow_html=True)  
    
    
elif choose == "Résultats":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Résultats</p>', unsafe_allow_html=True)

#   st.subheader('Tableau des metrics')
    st.markdown('Tableau de comparaison des metrics des différents modèles')
    
elif choose == "Conclusion":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Conclusion</p>', unsafe_allow_html=True)
    
    st.markdown("""
            Nous avons abordé ce projet sous un angle "plutôt" scolaire et en mode "laboratoire".<br>
            Le modèle qui a obtenu les meilleurs résultats est **la Régression Linéaire**.<br>
            <br>
            Pour aller plus loin, il faudrait aborder ce projet selon d'autres axes, par exemple :
            - Timeseries (qui n'est pas dans le cursus DataAnalyst)
            - Ajouter données externes (variables explicatives)
                - Température / Climat par région
                - Informations relatives à la population par région
                - Informations / indices sur le niveau d'industrialisation par région
            Il aurait aussi été intéressant de croiser ce projet avec celui d'un autre groupe (par exemple le projet Earth Temperature)
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()