import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Création d'un générateur de données pour la préparation des images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Préparation des données d'entraînement
train_generator = datagen.flow_from_directory(
        './dataset/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training')

# Préparation des données de validation
validation_generator = datagen.flow_from_directory(
        './dataset/Validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation')

# Création du modèle
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)), # 3 canaux pour les couleurs
        layers.MaxPooling2D(pool_size=(2, 2)), # Réduction de la taille de l'image
        layers.Conv2D(64, (3, 3), activation="relu"), # 64 filtres de convolution
        layers.MaxPooling2D(pool_size=(2, 2)), # Réduction de la taille de l'image
        layers.Conv2D(128, (3, 3), activation="relu"), # 128 filtres de convolution
        layers.MaxPooling2D(pool_size=(2, 2)), # Réduction de la taille de l'image
        layers.Conv2D(128, (3, 3), activation="relu"), # 128 filtres de convolution
        layers.MaxPooling2D(pool_size=(2, 2)), # Réduction de la taille de l'image
        layers.Flatten(), # Conversion de la matrice en vecteur
        layers.Dense(512, activation="relu"), # 512 neurones
        layers.Dense(1, activation="sigmoid"), # 1 neurone pour la prédiction
    ]
)

# Compilation du modèle
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Entraînement du modèle
model.fit(
        train_generator, # Données d'entraînement
        steps_per_epoch=2000, # Nombre d'images par époque
        epochs=10, # Nombre d'époques
        validation_data=validation_generator,# Données de validation
        validation_steps=800) # Nombre d'images par époque de validation

# Sauvegarde du modèle
model.save('model.h5')
