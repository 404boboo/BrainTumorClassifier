import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow import keras
import os
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint



DATASET_PATH = "mri/Training"
TESTING_PATH = "mri/Testing"
DEFAULT_BATCH_SIZE = 32
DEFAULT_WIDTH = 180
DEFAULT_HEIGHT = 180
EPOCHS = 60




def log_augmented_images(dataset):
    augmented_images, labels = next(iter(dataset.take(1)))
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axes[i].imshow(augmented_images[i].numpy().astype("uint8"))
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")
    wandb.log({"examples": fig})


# Split training data
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=10,
  image_size=(DEFAULT_HEIGHT, DEFAULT_WIDTH),
  batch_size=DEFAULT_BATCH_SIZE)

# Split validation data 
val_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="validation",
  shuffle=True,
  seed=10,
  image_size=(DEFAULT_HEIGHT, DEFAULT_WIDTH),
  batch_size=DEFAULT_BATCH_SIZE)

# Test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TESTING_PATH,
    image_size=(DEFAULT_HEIGHT, DEFAULT_WIDTH),  
    batch_size=DEFAULT_BATCH_SIZE,              
    shuffle=False # No shuffle for test data
)


wandb.init( # Wandb for logging metrics
  project="BrainTumorClassifier",
  config={
    "epochs": EPOCHS,
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": 0.0007,
    "dropout_rate": 0.6
  }
)

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


data_augmentation = keras.Sequential(
  [
    layers.RandomRotation(0.1, input_shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1) 
  ]
)

model = models.Sequential([
data_augmentation,
tf.keras.layers.Rescaling(1./255), # Convert RGB channel values to 0,1
layers.Conv2D(16, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),

layers.Conv2D(32, (3, 3), activation='relu'),
layers.MaxPooling2D((2,2)),

layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),

layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),



layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(wandb.config.dropout_rate),
layers.Dense(num_classes)
])

log_augmented_images(train_ds)
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


os.makedirs("wandb_models", exist_ok=True)
model_checkpoint = WandbModelCheckpoint("wandb_models/best_model.keras",
                                        save_best_only = True,
                                        monitor = "val_loss",
                                        mode = "min")

model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  
    restore_best_weights=True
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=wandb.config.epochs, 
  callbacks=[WandbMetricsLogger(), model_checkpoint],# Disable unless you have older version of tensorflow and wandb
  )
test_loss, test_accuracy = model.evaluate(test_ds)  # TEST EVALUATION

wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy}) 
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

model.save("finalModel.keras") 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('result.png')

plt.figure(figsize=(10,10))
for image, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3,3, i+1)
    plt.imshow(image[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
  plt.show()
# visual_image("C:/Users/ahmed/OneDrive/Desktop/FPT_proto/previewTestpoints.png")