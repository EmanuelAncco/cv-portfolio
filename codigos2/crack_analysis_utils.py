import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from sklearn.cluster import KMeans


# Step 1: Preprocessing functions
def preprocess_image(image_path):
    """Load, resize, and preprocess an image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # ResNet50 input size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    normalized = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized


# Step 2: Data augmentation
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Mantén esto para separar 80%/20%
)


train_generator = data_gen.flow_from_directory(    r'D:\Python projectos 2025\CNN EMANUEL\archive\data\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = data_gen.flow_from_directory(    r'D:\Python projectos 2025\CNN EMANUEL\archive\data\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


# Step 3: Define the CNN model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


# Step 5: Clustering for crack region analysis
def analyze_crack(image_path, model, n_clusters=2):
    """Analyze crack regions in the image using k-means clustering."""
    preprocessed = preprocess_image(image_path)
    prediction = model.predict(np.expand_dims(preprocessed, axis=0))[0][0]

    if prediction > 0.5:
        print("Crack detected with confidence:", prediction)
        reshaped = preprocessed.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped)
        labeled_image = kmeans.labels_.reshape(preprocessed.shape)
        return labeled_image
    else:
        print("No crack detected.")
        return None


# Example usage
image_path = 'data/test/sample.jpg'
labeled_crack = analyze_crack(image_path, model)
if labeled_crack is not None:
    cv2.imshow('Labeled Crack', labeled_crack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()