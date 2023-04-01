from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

classes = {0: '0-800',1:'801-3200'}

import sys
sys.path.append('/home/kittikhun62/guts_video_visualization/static/models/model.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model = tf.keras.models.load_model('/home/kittikhun62/guts_video_visualization/static/models/model.h5')

def resize_and_overlay_image(img_path,static):
    # Load the black background image
    background_path = 'static/black_background.png'
    background = Image.open(background_path)
    
    # Load the input image and resize it to 75x75
    input_image = Image.open(img_path)
    input_image = input_image.resize((150, 150))

    # Calculate the center position of the black background
    x = (background.width - input_image.width) // 2
    y = (background.height - input_image.height) // 2
    
    # Overlay the input image on the black background
    background.paste(input_image, (x, y))

    # Save the combined image to the output path
    background.save(static)

def predict_image(img_path, imageSize):
    # Read the image and preprocess it
    if imageSize == '10':
        target_size = (150, 150)
    else:
        target_size = (int(imageSize), int(imageSize))
    if imageSize == '20':
        # Resize and overlay the input image on the black background
        resize_and_overlay_image(img_path, img_path)
        
        # Update the target size to the size of the combined image
        background = Image.open(img_path)
        target_size = background.size
    if imageSize == '5':
        img = Image.open(img_path)
        width, height = img.size
        left = (width - 150) / 2
        top = (height - 150) / 2
        right = (width + 150) / 2
        bottom = (height + 150) / 2
        img = img.crop((left, top, right, bottom))
        img.save(img_path)    
        
    img = image.load_img(img_path, target_size=target_size)   
    x = image.img_to_array(img)   
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model.predict(x)
    return classes[result.argmax()]

@app.route('/')
def index():
    return render_template('test_upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Read the uploaded image and save it to a temporary file
        file = request.files['image']
        img_path = 'static/processed_image.png'
        file.save(img_path)
        # Get image size from the form
        imageSize = request.form['imageSize']              
        # Predict the age
        BET_pred = predict_image(img_path, imageSize)      

        # Render the prediction result
        return render_template('upload_completed.html', prediction=BET_pred)



if __name__ == '__main__':
    app.run(debug=True, port=8080)