import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

# ** GPU setup code **
print("Devices available:", tf.config.list_physical_devices())

# Enable GPU memory growth to avoid preallocating all memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

# Load pre-trained MobileNetV2 model and define new classification layers
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(90, activation="softmax")(x)
model = models.Model(inputs=base_model.input, outputs=x)

# Create the final model
classifier = Model(inputs=base_model.input, outputs=x)

# Freeze all layers of the base MobileNetV2 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with categorical crossentropy (multiclasses dataset)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dataset_path = r"dataset_path"

# Define splits and add augmentations for better diversity
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize image to [0, 1]
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=40,
    brightness_range=[0.2, 1.8],  # Randomly change brightness
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    fill_mode='nearest',
    validation_split=0.1,
)

# Training set (90% of data)
training_set = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)


# Validation set (10% of data)
validation_set = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)


# Calculate steps per epoch
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = validation_set.samples // validation_set.batch_size


# Train the model with callbacks and dynamic batch sizes
classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=validation_set,
    validation_steps=validation_steps,
)

# Save the trained model
classifier.save('custom_MobileNetV2_model.h5')
print("Customized MobileNetV2 model saved successfully.")