import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = "mri\Training"
DEFAULT_BATCH_SIZE = 32
DEFAULT_WIDTH = 180
DEFAULT_HEIGHT = 180


# Split training data
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="training",
  image_size=(DEFAULT_HEIGHT, DEFAULT_WIDTH),
  batch_size=DEFAULT_BATCH_SIZE)

# Split validation data 
val_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="validation",
  image_size=(DEFAULT_HEIGHT, DEFAULT_WIDTH),
  batch_size=DEFAULT_BATCH_SIZE)


# def visual_image(image_path):
#     img = Image.open(image_path)


class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes: ", class_names)


# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#     plt.show()


model = models.Sequential([
tf.keras.layers.Rescaling(1./255), # Convert RGB channel values to 0,1
layers.Conv2D(32, (3, 3), activation='relu'),
layers.MaxPooling2D((2,2)),

layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),

layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),

layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(4)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds,
  epochs=3)

model.summary()

# visual_image("C:/Users/ahmed/OneDrive/Desktop/FPT_proto/previewTestpoints.png")