import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw


def create_color_palette_from_matrix(dominant_colors, palette_size=(300, 50)):
    palette = Image.new("RGB", palette_size)
    draw = ImageDraw.Draw(palette)

    width = palette_size[0] // len(dominant_colors) if len(dominant_colors) > 0 else 0

    for i, color in enumerate(dominant_colors):
        color = tuple(map(int, color))
        draw.rectangle([i * width, 0, (i + 1) * width, palette_size[1]], fill=color)

    return palette


def main():
    st.title("Color Palette and Segmentation Streamlit App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Reshape the image for KMeans
        X = np.array(image).reshape(-1, 3)

        # Train KMeans algorithm
        #num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=20, value=3)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)

        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Convert cluster centers to a list of tuples
        dominant_colors = [tuple(map(int, color)) for color in cluster_centers]

        # Create and display the color palette
        color_palette = create_color_palette_from_matrix(dominant_colors, palette_size=(300, 50))
        st.image(color_palette, caption="Color Palette", use_column_width=True)

        # Optionally, display the segmented image
        if st.checkbox("Visualize Segmented Image"):
            segmented_image = cluster_centers[kmeans.labels_].reshape(np.array(image).shape)
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)


main()