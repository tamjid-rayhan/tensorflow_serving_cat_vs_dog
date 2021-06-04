import tensorflow as tf
import numpy as np
import json
import requests

#Docker command to run the model :-
#docker run -t --rm -p 8501:8501 
#-v "C:\Users\Tamjid\Documents\projects\tensorflow_serving\pets:/models/pets/1" 
#-e MODEL_NAME=pets tensorflow/serving

SIZE = 128
MODEL_URI= 'http://localhost:8501/v1/models/pets:predict'
classes = ['Cat' , 'Dog']


def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size = (SIZE, SIZE)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis = 0)

    data = json.dumps({
        'instances':image.tolist()
    })
    response = requests.post(MODEL_URI, data = data.encode())
    result = json.loads(response.text)
    #{'predictions': [[0.985524118]]}
    prediction = result['predictions'][0][0]
    class_name = classes[int(prediction > 0.5)]
    return class_name


