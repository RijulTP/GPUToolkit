import numpy as np
from keras.models import load_model
import keras.utils as image

def predict_image(imagepath, classifier):
    predict = image.load_img(imagepath, target_size=(64, 64))
    predict_modified = image.img_to_array(predict)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis=0)
    result = classifier.predict(predict_modified)

    if result[0][0] >= 0.5:
        prediction = 'dog'
        probability = result[0][0]
        print("Probability = " + str(probability))
        print("Prediction = " + prediction)
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
        print("Probability = " + str(probability))
        print("Prediction = " + prediction)

# Load the trained model
loaded_classifier = load_model('trained_model.h5')

# Example usage
dog_image = "dog.jpg"
predict_image(dog_image, loaded_classifier)

cat_image = "cat.jpg"
predict_image(cat_image, loaded_classifier)
