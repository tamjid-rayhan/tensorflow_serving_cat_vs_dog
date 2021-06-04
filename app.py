import os
import inference

from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap

SIZE= 128

app = Flask(__name__)
Bootstrap(app)

"""
Routes
"""
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join('static', uploaded_file.filename)
                uploaded_file.save(image_path)
                class_name = inference.get_prediction(image_path)
                print(class_name)
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path,
                    'size': SIZE
                }
                return render_template('show.html' , result = result)


    return render_template("index.html")
    

if __name__ == '__main__':
    app.run(debug = True)
