import cv2
import tensorflow as tf

# Chargement du modèle
model = tf.keras.models.load_model('model.h5')

# Création des labels
labels = {0: 'Femme', 1: 'Homme'}

# Configuration de la caméra
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Largeur de la vidéo
cap.set(4, 480) # Hauteur de la vidéo

# Boucle infinie pour traiter chaque image de la caméra
while True:
    # Capture de l'image
    ret, frame = cap.read()

    # Redimensionnement de l'image pour correspondre aux dimensions d'entrée du modèle
    img = cv2.resize(frame, (224, 224))

    # Prétraitement de l'image pour la passer en entrée du modèle
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255
    img = tf.expand_dims(img, axis=0)

    # Prédiction du genre
    prediction = model.predict(img)[0][0]

    # Affichage de la prédiction sur l'image
    label = labels[int(prediction >= 0.5)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Affichage de l'image
    cv2.imshow('Gender Detection', frame)

    # Sortie de la boucle si la touche 'q' est appuyée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la caméra et fermeture de la fenêtre
cap.release()
cv2.destroyAllWindows()
