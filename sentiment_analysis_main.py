import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model





# Initializing the ImageDataGenerator, rescaling the images and defining the validation split


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.14,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'MMAFEDB/train',  # Directory of training images
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=32,  # Number of images to return in each batch
    class_mode='categorical',  # Return one-hot encoded labels
    color_mode='rgb',  # Use RGB color space
    shuffle=True,  # Shuffle the data
    seed=42  # Random seed for reproducibility
)

# Flow validation images in batches of 32 using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
    'MMAFEDB/valid',  # Directory of validation images
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=32,  # Number of images to return in each batch
    class_mode='categorical',  # Return one-hot encoded labels
    color_mode='rgb',  # Use RGB color space
    shuffle=True,  # Shuffle the data
    seed=42  # Random seed for reproducibility
)

# Flow test images in batches of 32 using train_datagen generator
test_generator = train_datagen.flow_from_directory(
    'MMAFEDB/test',  # Directory of test images
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=32,  # Number of images to return in each batch
    class_mode='categorical',  # Return one-hot encoded labels
    color_mode='rgb',  # Use RGB color space
    shuffle=True,  # Shuffle the data
    seed=42  # Random seed for reproducibility
)


"""
The classes used for the dataser are: digust, happy, neutral and surprise because they are more useful
in the contest of a marketing company who wants to detect the reactions to customers
to a new product
"""
class_names = ['disgust', 'happy', 'neutral', 'surprise']

"""
Function to define focal loss, and to handle class imbalance
"""
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()  # Small value to avoid division by zero
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions to avoid log(0)
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Calculate focal loss components
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

        return K.mean(fl)
    return focal_loss_fixed

# Define the model architecture to improve validation accuracy and reduce overfitting
model = models.Sequential([
    # First convolutional layer with batch normalization and l2 regularizers:
    layers.Conv2D(48, (5, 5), activation='relu', input_shape=(48, 48, 3), kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    #layers.MaxPooling2D((2, 2)),

    # Second convolutional layer with batch normalization and l2 regularizers:
    layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    #layers.MaxPooling2D((2, 2)),

    #Third convolutional layer with batch normalization, l2 regularization and max pooling:
    layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    #Fourth convolutional layer with batch normalization, l2 regularizers, dropout and max pooling:
    layers.Conv2D(256, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Flatten the output from convolutional layers
    layers.Flatten(),
    # First fully connected dense layer with batch normalization and dropout
    layers.Dense(2000, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    # Second fully connected dense layer with batch normalization:
    layers.Dense(1000, activation='relu'),
    layers.BatchNormalization(),
    # Output layer with softmax activation for multi-class classification
    layers.Dense(len(class_names), activation='softmax')
])


"""
This code initializes two callbacks for training a neural network model:

EarlyStopping: monitors the validation loss and stops training if it doesn't improve for a specified number of epochs.
    - monitor: 'val_loss' specifies that validation loss is being monitored.
    - patience: Tested with 10 epochs with no improvement, afterwards, the training will be stopped.
    - restore_best_weights: It is set to True, therefore, the model weights will be restored to the state of the epoch
      with the best validation loss.

ReduceLROnPlateau: reduces the learning rate when the validation loss has stopped improving.
    - monitor: 'val_loss' specifies that validation loss is being monitored.
    - factor: The factor by which the learning rate will be reduced. new_lr = lr * factor.
    - patience: in this case 5 epochs with no improvement, then the learning rate will be reduced.
    - min_lr: The lower bound on the learning rate.
"""
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

"""
Function to calculate the recall performance of the model
"""
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

"""
Function to calculate the precision of the model
"""
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

"""
Function to show the performance of the model with F1 score
"""
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#Compile the model with the custom focal loss and custom metrics
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=focal_loss(gamma=2., alpha=0.25),
              metrics=['accuracy', precision_m, recall_m, f1_m])

# Train the model on the training data with validation#
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on a single batch from the test set
test_images, test_labels = next(test_generator)
results = model.evaluate(test_images, test_labels)
test_loss = results[0]
test_acc = results[1]

# Print the test loss and accuracy outcome
print('TEST LOSS:', test_loss)
print('TEST ACCURACY:', test_acc)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Metrics')
plt.ylabel('Metric Value')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# Plot all other metrics: precision, recall, and F1 score
plt.plot(history.history['precision_m'], label='Train precision')
plt.plot(history.history['val_precision_m'], label='Validation precision')
plt.plot(history.history['recall_m'], label='Train recall')
plt.plot(history.history['val_recall_m'], label='Validation recall')
plt.plot(history.history['f1_m'], label='Train F1 Score')
plt.plot(history.history['val_f1_m'], label='Validation F1 Score')
plt.title('Model Metrics')
plt.ylabel('Metric Value')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Show a summary of the model
model.summary()

# Get a batch of images
images, labels = next(train_generator)

# Because the generator performs rescaling, we need to rescale back to display properly
images = images * 255

# Predict on the test images
predictions = model.predict(test_images)
p0 = predictions[0]
print(p0)
print(predictions.shape)

# Find the class with the highest probability for the first image
highest = 0
idx = -1
for ix, probability in enumerate(p0):
    if probability > highest:
        highest = probability
        idx = ix
print(class_names[idx])


"""
Function to plot images in a 4x4 grid with labels from: happy, disgust, surprise, neutral
"""
def plot_images(images_arr, labels):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for img, label, ax in zip(images_arr, labels, axes):
        img_rescaled = img.astype('uint8')  # Convert the image back to uint8
        ax.imshow(img_rescaled)
        ax.set_title(class_names[np.argmax(label)])  # Set the label as title
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plotting the first 16 images in the batch to see the outcome
plot_images(images[:16], labels[:16])


# showing the model used in a graphical rapresentation
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# using Grapbiz for the visualisation of the model in a clearer way.
try:
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print(f"Error in plotting model: {e}")
