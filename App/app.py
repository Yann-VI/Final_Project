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
import plotly.express as px

def main():
    with st.sidebar:
        st.sidebar.header(":blue_car: Projet Wake Up")
        choose = option_menu("Menu App", ["Contexte","Montres ta tête !", "Balances ta cam", "Stream"],
                            icons=['house','camera fill','camera-video-fill'],
                            menu_icon="cast", default_index=1,
                            styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
            
        }
        )

    if choose == "Contexte":
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Contexte</p>', unsafe_allow_html=True)    
        
        st.image('vbt.jpg')

        st.markdown("""
                Le but de ce projet est de pouvoir déceler **un comportement d'endormissement**.<br>
                Pour réaliser cette étude nous avons mis en place cette application qui permet à l'utilisateur de simuler son état de somnolence.<br> 
                Les options proposées via notre application sont les suivantes : <br>
                - Télécharger directement une photo depuis son PC.<br>
                - Télécharger une photo depuis sa webcam.<br>
                """, unsafe_allow_html=True)

    elif choose == "Montres ta tête !" :
        
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Contexte</p>', unsafe_allow_html=True)
        
        st.title("Image Loader")

        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the image here (e.g., save it to a specific location)
            #image = uploaded_file.getbuffer()
            #file_path = save_uploaded_file_to_temp(uploaded_file)
            #st.write(file_path)
            #save_image_to_s3(file_path, 'temp/test.jpg')
            st.success("Image loaded successfully!")
            st.image(uploaded_file)

            if st.button("Prediction"):
                # Send request to FastAPI server
                #file_name = "uploaded_image.png"
                api_url = "http://host.docker.internal:4000/predict"  # Replace with your FastAPI server URL
                data = {"file": uploaded_file.getvalue(), "type":"image/jpeg"}
                response = requests.post(api_url, files=data)

                if response.status_code == 200:
                    result = response.json()
                    st.image(np.array(result["image"]))
                    st.write("Prediction Result:", result['response'])
                else:
                    st.image(uploaded_file)
                    st.write("Prediction Failed!")


    elif choose == "Stream":

        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Contexte</p>', unsafe_allow_html=True)

        st.title("Image Loader")

        # File uploader
        uploaded_file = st.camera_input("Take a picture")

        if uploaded_file is not None:
            # Process the image here (e.g., save it to a specific location)
            st.success("Image loaded successfully!")
            

            if st.button("Prediction"):
                # Send request to FastAPI server
                api_url = "http://host.docker.internal:4000/predict"
                data = {"file": uploaded_file.getvalue(), "type":"image/jpeg"}
                response = requests.post(api_url, files=data)

                if response.status_code == 200:
                    result = response.json()
                    st.image(np.array(result["image"]))
                    st.write("Prediction Result:", result['response'])
                else:
                    st.image(uploaded_file)
                    st.write("Prediction Failed!")

    elif choose == "Balances ta cam":

        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Contexte</p>', unsafe_allow_html=True)

        st.title("Image Loader")

        # File uploader
        uploaded_file = st.camera_input("Take a picture")

        if uploaded_file is not None:
            # Process the image here (e.g., save it to a specific location)
            st.success("Image loaded successfully!")

            if st.button("Prediction"):
                # Send request to FastAPI server
                api_url = "http://host.docker.internal:4000/predict"
                data = {"file": uploaded_file.getvalue(), "type":"image/jpeg"}
                response = requests.post(api_url, files=data)

                if response.status_code == 200:
                    result = response.json()
                    st.image(np.array(result["image"]))
                    st.write("Prediction Result:", result['response'])
                else:
                    st.image(uploaded_file)
                    st.write("Prediction Failed!")

if __name__ == '__main__':
    main()