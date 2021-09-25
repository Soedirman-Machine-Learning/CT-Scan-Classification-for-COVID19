from __future__ import division, print_function

# coding=utf-8
import sys
import glob
import re
import os
from flask import Flask, render_template, Response, url_for, redirect
from flask import request

# Keras
from keras.preprocessing import image
from keras.models import load_model
import h5py

# TF
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

app =  Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH_LP = os.path.join(BASE_PATH, "static/models", "LP.h5")
MODEL_PATH_CP = os.path.join(BASE_PATH, "static/models", "CP.h5")
MODEL_PATH_RISK = os.path.join(BASE_PATH, "static/models", "risk.h5")
MODEL_PATH_MOR = os.path.join(BASE_PATH, "static/models", "mor.h5")

model_LP = load_model(MODEL_PATH_LP)
model_CP = load_model(MODEL_PATH_CP)
model_risk = load_model(MODEL_PATH_RISK)
model_mor = load_model(MODEL_PATH_MOR)
print("Model loaded")


def model_predict_LP(img_path, model):
    classes_LP = ["NiCT", "nCT", "pCT"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    confidence = np.amax(result)
    text = str(classes_LP[np.argmax(result)])
    return text, confidence
    # for i in result:
    #     confidence = np.amax(i)
    #     text = classes_LP[np.argmax(i)]
    #     return text, confidence


def model_predict_CP(img_path, model):
    classes_CP = ["Negative", "Positive"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    confidence = np.amax(result)
    text = str(classes_CP[np.argmax(result)])
    return text, confidence


def model_predict_risk(img_path, model):
    classes_risk = ["Control", "Type I", "Type II"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    confidence = np.amax(result)
    text = str(classes_risk[np.argmax(result)])
    return text, confidence


def model_predict_mor(img_path, model):
    classes_mor = ["Cured", "Deceased", "Unknown"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    confidence = np.amax(result)
    text = str(classes_mor[np.argmax(result)])
    return text, confidence

@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page not found. Please go the home page and try again"
    return render_template("error.html", message=message)

@app.errorhandler(405)
def error405(error):
    message = "Error 405, Method not found"
    return render_template("error.html",message=message)
  
@app.errorhandler(500)
def error500(error):
    message = "INTERNAL ERROR 500, Error occurs in the program"
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            pred_type = 0
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved succesfully')
            # Make prediction
            preds_LP, conf_LP = model_predict_LP(path_save, model_LP)
            conf_LP = int(round(conf_LP, 4) * 100)

            preds_CP, conf_CP = model_predict_CP(path_save, model_CP)
            conf_CP = int(round(conf_CP, 4) * 100)

            preds_risk, conf_risk = model_predict_risk(path_save, model_risk)
            conf_risk = int(round(conf_risk, 4) * 100)

            preds_mor, conf_mor = model_predict_mor(path_save, model_mor)
            conf_mor = int(round(conf_mor, 4) * 100)

            if preds_LP == "pCT" and preds_CP == "Positive":
                pred_type = 3
            elif preds_LP == "pCT" and preds_CP == "Negative":
                pred_type = 2
            elif preds_LP != "pCT":
                pred_type = 1
            else:
                pred_type = 0

            print("Prediksi: ", preds_mor)
            print("Probabilitas: ", conf_mor, "%")
            return render_template("upload.html",
                prediction_LP=preds_LP,
                confidence_LP=conf_LP,
                prediction_CP=preds_CP,
                confidence_CP=conf_CP,
                prediction_risk=preds_risk,
                confidence_risk=conf_risk,
                prediction_mor=preds_mor,
                confidence_mor=conf_mor,
                extension=False,
                fileupload=True,
                prediction_type=pred_type,
                image_filename=filename,
                )
           
        else:
            print('Use only the extension with .jpg, .png, .jpeg ')
            return render_template('upload.html', extension=True,fileupload=False)

    else: 
        return render_template('upload.html',fileupload=False,extension=False)

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)