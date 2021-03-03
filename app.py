# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:17:58 2021

@author: keerthi
"""


from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
#from ./monkey_pred import predict


app = Flask(__name__, template_folder="template")

label = {0:'mantled_howler', 1:'patas_monkey', 2:'bald_uakari', 3:'japanese_macaque', 4:'pygmy_marmoset', 
         5:'white_headed_capuchin', 6:'silvery_marmoset', 7:'common_squirrel_monkey',
         8:'black_headed_night_monkey', 9:'nilgiri_langur'} 

model = load_model('D:/internship/img_clsf/inceptionv3.h5')
path = 'D:/internship/monkeys_species/validation/validation/n3/n317.jpg'

@app.route("/", methods=["GET"])
def home():
    #load html page
    #print(predict('D:/internship/monkeys_species/validation/validation/n3/n317.jpg'))
    return render_template("index.html")


def predict(path):
  img = load_img(path, color_mode='rgb',target_size=(224,224))
  img = img_to_array(img)
  img = img/255.
  img = np.array([img])
  prediction = model.predict(img)
  prediction = np.argmax(prediction)
  return label[prediction]


@app.route("/submit", methods=["POST"])

def deploy():
    image = request.files["my_img"]
    path = "static/" + image.filename
    image.save(path)
    
    pred = predict(path)
    
    return render_template("index.html", prediction = pred, img_path = path)

if __name__=='__main__':
    app.run(debug=True)

    
    
    
