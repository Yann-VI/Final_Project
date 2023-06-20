import streamlit as st
import boto3
import os
import requests
import botocore
from io import BytesIO


""" S3_BUCKET = "wakeup-jedha"

def save_image_to_s3(filename, name_output):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ID_KEY,aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
        with open(filename, 'rb') as file:
            s3_client.upload_fileobj(file, S3_BUCKET, name_output)
        os.remove(filename)
        return True
    except botocore.exceptions.ClientError as e:
        st.error(f"Error uploading image to S3: {e}")
        return False
    """
# Save the uploaded file to a temporary file before uploading to S3
def save_uploaded_file_to_temp(uploaded_file):
    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return uploaded_file.name
    except Exception as e:
        st.exception("Failed to save file.")
        return None



def main():
    st.title("Image Loader")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    #uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is not None:
        # Process the image here (e.g., save it to a specific location)
        #image = uploaded_file.getbuffer()
        file_path = save_uploaded_file_to_temp(uploaded_file)
        st.write(file_path)
        #save_image_to_s3(file_path, 'temp/test.jpg')
        st.success("Image loaded successfully!")
        st.image(uploaded_file)

        if st.button("Prediction"):
            # Send request to FastAPI server
            file_name = "uploaded_image.png"
            api_url = "http://host.docker.internal:4000/predict"  # Replace with your FastAPI server URL
            data = {"file": uploaded_file.getvalue(), "type":"image/jpeg"}
            response = requests.post(api_url, files=data)

            if response.status_code == 200:
                result = response.json()
                st.write("Prediction Result:", result)
            else:
                st.write(response.content)
                st.write("Prediction Failed!")





if __name__ == '__main__':
    main()