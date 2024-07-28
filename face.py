import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Charger le modèle de détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fonction pour détecter les visages dans une image
def detect_faces(image, scale_factor, min_neighbors):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return faces

# Interface utilisateur Streamlit
st.title("Application de Détection de Visages")
st.write("""
Cette application utilise l'algorithme Viola-Jones pour détecter les visages dans une image.
Téléchargez une image et ajustez les paramètres pour améliorer la détection.
""")

# Ajouter un champ pour télécharger une image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lire l'image téléchargée
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Sélection de la couleur du rectangle
    color = st.color_picker('Choisissez la couleur du rectangle', '#00FF00')
    color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Ajustement des paramètres de détection
    scale_factor = st.slider('Facteur d\'échelle (scaleFactor)', 1.1, 2.0, 1.1, 0.1)
    min_neighbors = st.slider('Nombre minimum de voisins (minNeighbors)', 1, 10, 5)

    # Détection des visages
    if st.button('Détecter les visages'):
        faces = detect_faces(image, scale_factor, min_neighbors)
        
        # Dessiner des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        st.image(image, caption='Image avec visages détectés', use_column_width=True)
        
        # Enregistrer l'image avec les visages détectés
        if st.button('Enregistrer l\'image avec visages détectés'):
            save_path = 'detected_faces.jpg'
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            st.success(f'Image enregistrée sous {save_path}')
else:
    st.write("Veuillez télécharger une image pour la détection des visages.")
