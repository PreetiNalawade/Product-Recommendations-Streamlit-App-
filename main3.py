import streamlit as st
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messages

from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Product Recommendation System')

def save_uploaded_file(uploaded_file):
    try:
        # Create 'uploads' directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)

        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result
styles_df = pd.read_csv('styles.csv')
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# File upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommendation
        indices = recommend(features, feature_list)
        # Show
        col1, col2, col3, col4, col5 = st.columns(5)

        def display_image_in_column(column, index):
            file_path = filenames[indices[0][index]].replace("\\", "/")  # Replace backslashes with forward slashes
            
            # Fetch image details from styles_df
            image_id = os.path.basename(file_path).split('.')[0]
            
            # Check if image_id exists in styles_df
            if int(image_id) in styles_df['id'].values:
                image_details = styles_df[styles_df['id'] == int(image_id)].iloc[0]

                with column:
                    st.image(file_path)
                    st.caption(f"Product ID: {image_id}")
                    st.write(f"**Name:** {image_details['productDisplayName']}")
                    st.write(f"**Category:** {image_details['masterCategory']}")
                    st.write(f"**Subcategory:** {image_details['subCategory']}")
                    st.write(f"**Description:** {image_details['description']}")
                    st.write(f"**Price:** {image_details['price']} {image_details['currency']}")
            else:
                with column:
                    st.image(file_path)
                    st.caption(f"Product ID: {image_id}")
                    st.warning("No information found for this product.")

        display_image_in_column(col1, 0)
        display_image_in_column(col2, 1)
        display_image_in_column(col3, 2)
        display_image_in_column(col4, 3)
        display_image_in_column(col5, 4)
    else:
        st.header("Some error occurred in file upload")
