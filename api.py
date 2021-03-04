from flask import Flask,render_template
import os
from flask import request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def model_predict(image_location):
    img = image.load_img(image_location,target_size = (150,150))
    model = load_model('vgg16_nature_classifier1.h5')
    img = image.img_to_array(img)
    img = img/255
    test_img = np.expand_dims(img,axis=0)
    prediction = np.argmax(model.predict(test_img),axis=1)[0]
    d = {0 : 'Buildings',1 :  'Forest' , 2 : 'Glacier', 3 : 'Mountain',  4 : 'Sea', 5 : 'Street'}
    output = d[prediction]
    return output







upload_folder = "upload"


app=Flask(__name__,template_folder='templates') #__name__ is a built-in variable which evaluates to the name of the current module. Thus it can be used to check whether the current script is being run on its own or being imported somewhere else by combining it with if statement, as shown below.
#Every Python module has it's __name__ defined and if this is '__main__',
#it implies that the module is being run standalone by the user and we can do corresponding appropriate actions. ... if __name__ == “main”: is used to execute some code only if the file was run directly, and not imported.

@app.route("/",methods = ["GET","POST"]) #/-->local server, methods--> getting the output,posting the images
def predict():
    if request.method == "POST": #this is where the request to post is done
        image_file = request.files["image"]
        if image_file: #if something has been uploaded
            image_location = os.path.join(upload_folder,image_file.filename) #we save it
            image_file.save(image_location)
            result =  model_predict(image_location)
            return render_template('index.html',prediction = "This is an image of " + result + " !")#if the above operation is successful, render t

    return render_template('index.html',prediction = 'Upload the pic')


if __name__=="__main__":
    app.run(debug=True)
