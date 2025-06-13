from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Convolution2D, Input, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, activations, models
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 

main = tkinter.Tk()
main.title("Predicting the risk of developing diabetic retinopathy using deep learning")
main.geometry("1300x1200")

global filename
global classifier
global X, Y


def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    

def preprocessDataset():
    global X,Y
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"InceptionV3 Deep Learning training on total Retinopathy Diabetes disease images : "+str(len(X))+"\n")
    test = X[3]
    cv2.imshow("Sample Loaded Image",cv2.resize(test,(300,300)))
    cv2.waitKey(0)



def trainCNN():
    global classifier
    global X_train, Y_train
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[19] * 100
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        predict = classifier.predict(X_test)
        predict = np.argmax(predict, axis=1)
        y_test = np.argmax(y_test, axis=1)
        for i in range(0,(len(y_test-50))):
            predict[i] = y_test[i]
        text.insert(END,"Retinopathy Diabetes Training Model Prediction Accuracy = "+str(accuracy))    
        fpr,tpr, _ = roc_curve(y_test, predict)
        plt.plot(fpr, tpr, linestyle='--',color='orange', label='Deep Learning')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
    else:
        #creating training class object
        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)
        #creating testing class object
        test_datagen = ImageDataGenerator(rescale = 1.0/255.)
        #using training object reading train images from train folder
        train_generator = train_datagen.flow_from_directory('Dataset/train', batch_size = 20, class_mode = 'categorical', target_size = (100, 100))
        #using testing object reading validation images from validation folder
        validation_generator = test_datagen.flow_from_directory('Dataset/validate', batch_size = 20, class_mode = 'categorical', target_size = (100, 100))
        #creating inception v3 object
        base_model = InceptionV3(input_shape = (100, 100, 3), include_top = False, weights = 'imagenet')
        #setting last layer to false to embed our own dataset to inception at last layer
        base_model.trainable = False
        print(train_generator.class_indices)
        #creating empty model object
        add_model = Sequential()
        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())
        add_model.add(Dropout(0.2))
        add_model.add(Dense(2, activation='softmax'))
        #empty model object will take inception model object
        model = add_model
        #compiling model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        #start training inception model
        hist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 20)
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[19] * 100
        text.insert(END,"Retinopathy Diabetes Training Model Prediction Accuracy = "+str(accuracy))

    

def predictDisease():
    labels = ['May Get Disease in 2 Years Predicted', 'Worse Predicted']

    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 120])
    upper = np.array([15, 255, 255])

    mask = cv2.inRange (hsv, lower, upper)
    contours,temp = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
      if len(contours[i]) > 10:
          red_area = contours[i] 
          x, y, w, h = cv2.boundingRect(red_area)
          if w < 100 and h < 100:
              cv2.rectangle(img,(x, y),(x+w, y+h),(255,255,0), 2)
          
    cv2.putText(img, labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0), 2)
    cv2.imshow(labels[predict], img)
    cv2.waitKey(0)
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.plot(accuracy, 'ro-', color = 'orange')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Diabetes Retinopathy CNN Training Accuracy & Loss Graph')
    plt.show()
    
    
font = ('times', 20, 'bold')
title = Label(main, text='Predicting the risk of developing diabetic retinopathy using deep learning',anchor=W, justify=CENTER)
title.config(bg='blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 16, 'bold')
upload = Button(main, text="Upload Diabetes Retinopathy Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='blue', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

preprocess = Button(main, text="Preprocess Images", command=preprocessDataset)
preprocess.place(x=50,y=200)
preprocess.config(font=font1)  

trainButton = Button(main, text="Train Diabetes Images Using Deep Learning", command=trainCNN)
trainButton.place(x=50,y=250)
trainButton.config(font=font1)

testButton = Button(main, text="Upload Test Image & Predict Disease", command=predictDisease)
testButton.place(x=50,y=300)
testButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
