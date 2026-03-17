import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your trained model
model = tf.keras.models.load_model('pneumonia_resnet50.h5')

# Paths
test_dir = 'dataset/test'

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")
