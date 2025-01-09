import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

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
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)  # Added dropout
x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)  # Add regularizer
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
    rescale=1./255,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2], 
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    validation_split=0.2,
)

# Training set (80% of data)
training_set = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)


# Validation set (20% of data)
validation_set = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
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