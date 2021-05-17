# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:17:58 2021

@author: keerthi
"""


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, template_folder="template")

label = {0:'Mantled Howler', 1:'Patas Monkey', 2:'Bald Uakari', 3:'Japanese Macaque', 4:'Pygmy Marmoset', 
         5:'White Headed Capuchin', 6:'Silvery Marmoset', 7:'Common Squirrel Monkey',
         8:'Black Headed Night Monkey', 9:'Nilgiri Langur'} 

char = {0:['*Mantled howler: The mantled howler, or golden-mantled howling monkey, is a species of howler monkey, a type of New World monkey, from Central and South America. It is one of the monkey species most often seen and heard in the wild in Central America. It takes its "mantled" name from the long guard hairs on its sides.','Scientific name: Alouatta palliata','Kingdom:Animalia','Phylum: Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Atelidae','Genus:Alouatta'],
        1:['*Patas monkey: The patas monkey, also known as the wadi monkey or hussar monkey, is a ground-dwelling monkey distributed over semi-arid areas of West Africa, and into East Africa.','Scientific name: Erythrocebus patas','Higher classification: Erythrocebus','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family: Cercopithecidae'],
        2:['*Bald uakari:The bald uakari or bald-headed uakari is a small New World monkey characterized by a very short tail; bright, crimson face; a bald head; and long coat. The bald uakari is restricted to várzea forests and other wooded habitats near water in the western Amazon of Brazil and Peru.','Scientific name: Cacajao calvus','Higher classification: Uakaris','Kingdom:Animalia','Phylum:	Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family: Pitheciidae'],
        3:['*Japanese macaque: The Japanese macaque, also known as the snow monkey, is a terrestrial Old World monkey species that is native to Japan. They get their name "snow monkey" because some live in areas where snow covers the ground for months each year – no other non-human primate is more northern-living, nor lives in a colder climate.','Scientific name: Macaca fuscata','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Cercopithecidae'],
        4:['*Pygmy marmoset: The pygmy marmoset, genus Cebuella, is a small genus of New World monkey native to rainforests of the western Amazon Basin in South America. It is notable for being the smallest monkey and one of the smallest primates in the world, at just over 100 grams.\n\nScientific name: Cebuella pygmaea','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Callitrichidae'],
        5:['*White headed capuchin: The Panamanian white-faced capuchin, also known as the Panamanian white-headed capuchin or Central American white-faced capuchin, is a medium-sized New World monkey of the family Cebidae, subfamily Cebinae.','Scientific name: Cebus imitator','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Cebidae','Genus:Cebus'],
        6:['*Silvery marmoset: The silvery marmoset is a New World monkey that lives in the eastern Amazon Rainforest in Brazil. The fur of the silvery marmoset is colored whitish silver-grey except for a dark tail. Remarkable are its naked, flesh-colored ears which stand out from the skin.','Scientific name: Mico argentatus','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Callitrichidae','Genus:	Mico'],
        7:['*Common squirrel monkey:Common squirrel monkey is the traditional common name for several small squirrel monkey species native to the tropical areas of South America.','Scientific name: Saimiri sciureus','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:Cebidae','Subfamily:Saimiriinae'],
        8:['*Black headed night monkey: The black-headed night monkey is a night monkey species from South America. It is found in Bolivia, Brazil and Peru. The A. nigriceps in Peru were notably inhabiting areas that were degraded, and often these areas were disturbed either by human activities or natural occurrences in the ecosystem.','Scientific name: Aotus nigriceps','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:	Aotidae','Genus:Aotus'],
        9:['*Nilgiri langur: The Nilgiri langur is a langur found in the Nilgiri Hills of the Western Ghats in South India. Its range also includes Kodagu in Karnataka, Kodayar Hills in Tamil Nadu, and many other hilly areas in Kerala and Tamil Nadu. This primate has glossy black fur on its body and golden brown fur on its head.','Scientific name: Trachypithecus johnii','Kingdom:Animalia','Phylum:Chordata','Class:Mammalia','Order:Primates','Suborder:Haplorhini','Infraorder:Simiiformes','Family:	Cercopithecidae','Genus:Semnopithecus']}

model = tf.keras.models.load_model('D:/internship/img_clsf/inceptionv3.h5')
 
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


def predict(path):
    img = load_img(path, color_mode='rgb',target_size=(224,224))
    img = img_to_array(img)
    img = img/255.
    img = np.array([img])
    
    prediction = model.predict(img)
    prediction = np.argmax(prediction)
    
    return label[prediction], char[prediction]



@app.route("/submit", methods=["POST"])

def deploy():
    image = request.files["my_img"]
    path = "static/" + image.filename
    image.save(path)
    
    pred, text = predict(path)
    
    return render_template("index.html", prediction = pred, img_path = path, content = text)

if __name__=='__main__':
    app.run(debug=True, use_reloader=False)

    
    
    
