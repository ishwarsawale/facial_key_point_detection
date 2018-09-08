'''
MIT License

Copyright (c) [2018] [Ishwar Sawale]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import time 
import cv2
import time 
import utils
from keras.models import load_model

# Import deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Input
from keras.models import Model

def resize_images(img):
    """Returns resized image
    Cannot be directly used in lambda function
    as tf is not understood by keras
    """
    import tensorflow as tf
    return tf.image.resize_images(img, (96, 96))
def wrapper_resize_image(model):
  inp = Input(shape=(None, None, 1))
  resize_inp = Lambda(resize_images)(inp)
  out = model(resize_inp)
  model_with_resize = Model(input=inp, output=out)
  return model_with_resize
def deal_with_image(image_name,model):
  # Load in color image for face detection
  model_with_resize = wrapper_resize_image(model)
  image = cv2.imread(image_name)
  # Convert the image to RGB colorspace
  image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.25, 6)

  fig = plt.figure(figsize = (9,9))
  ax1 = fig.add_subplot(111)
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1.set_title('image copy')

  for (x,y,w,h) in faces:
      cv2.rectangle(image_copy, (x,y), (x+w,y+h), (255,0,0), 3)
      gray_roi = gray[y:y+h, x:x+w].reshape(1, h, w, 1)/255.0
      y_roi = model_with_resize.predict(gray_roi)
      landmarks_x = x + (y_roi[0][0::2] + 1) * w/2
      landmarks_y = y + (y_roi[0][1::2] + 1) * h/2
      ax1.scatter(landmarks_x, landmarks_y, marker='o', c='#19dc26', s=10)
  ax1.imshow(image_copy)
  plt.show()
def live_AllPoints(model):
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    model_with_resize = wrapper_resize_image(model)
    vc = cv2.VideoCapture(0)
    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    # keep video stream open
    while rval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
            gray_roi = gray[y:y+h, x:x+w].reshape(1, h, w, 1)/255.0
            y_roi = model_with_resize.predict(gray_roi)
            landmarks_x = x + (y_roi[0][0::2] + 1) * w/2
            landmarks_y = y + (y_roi[0][1::2] + 1) * h/2
            for pt in zip(landmarks_x, landmarks_y):
                cv2.circle(frame, pt, 1, (255,0,0), -1)
        # plot image from camera with detections marked
        cv2.imshow("face detection activated", frame)
        # exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # exit by pressing any key
            # destroy windows
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)
            return
        # read next 63frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()  
def plot_points(image_name,model):
  # Load in color image for face detection
  model_with_resize = wrapper_resize_image(model)
  image = cv2.imread(image_name)
  # Convert the image to RGB colorspace
  image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.25, 6)
  for (x,y,w,h) in faces:
      cv2.rectangle(image_copy, (x,y), (x+w,y+h), (255,0,0), 3)
      gray_roi = gray[y:y+h, x:x+w].reshape(1, h, w, 1)/255.0
      y_roi = model_with_resize.predict(gray_roi)
      landmarks_x = x + (y_roi[0][0::2] + 1) * w/2
      landmarks_y = y + (y_roi[0][1::2] + 1) * h/2
      all_points = zip(landmarks_x,landmarks_y)
      for idx, point in enumerate(all_points):
        cv2.putText(image, str(idx), point,
                                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                fontScale=0.4,
                                color=(0, 0, 255))
        cv2.circle(image, point, 3, color=(0, 255, 255))
  cv2.imwrite("image_with_point_drawn.png", image)
  # cv2.imshow('Result', image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)
def get_model():
  model = Sequential()
  model = Sequential()
  model.add(Conv2D(16, (3, 3), border_mode='valid', input_shape=(96, 96, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2))) 
  model.add(Conv2D(32, (3, 3), border_mode='valid', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2))) 
  model.add(Conv2D(64, (3, 3), border_mode='valid', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(250, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(30, activation='linear'))
  # model.summary()
  return(model)
def train_models(model,X_train,y_train,X_test):
  from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
  from keras import backend as K

  optimizers = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
  for x in optimizers:
      opt = str(x.__name__)
      print("\n\n" + opt)
      session = K.get_session()
      for layer in model.layers: 
          for v in layer.__dict__:
              v_arg = getattr(layer,v)
              if hasattr(v_arg,'initializer'):
                  initializer_method = getattr(v_arg, 'initializer')
                  initializer_method.run(session=session)
                  ##print('reinitializing layer {}.{}'.format(layer.name, v))
      ## Compile the model
      model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
      from keras.callbacks import ModelCheckpoint  
      checkpointer = ModelCheckpoint(filepath='saved_models/' + opt + '.h5', verbose=1, save_best_only=True)
      ## Train the model
      exec('hist_' + str(opt) + '= model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[checkpointer], verbose=0)')

def get_data():
  from utils import *
  import matplotlib.pyplot as plt
  # Load training set
  X_train, y_train = load_data()
  print("X_train.shape == {}".format(X_train.shape))
  print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
      y_train.shape, y_train.min(), y_train.max()))
  # Load testing set
  X_test, _ = load_data(test=True)
  print("X_test.shape == {}".format(X_test.shape))
  fig = plt.figure(figsize=(20,20))
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
  index = 0
  for i in range(10, 19):
      ax = fig.add_subplot(3, 3, index + 1, xticks=[], yticks=[])
      plot_data(X_train[i], y_train[i], ax)
      index +=1
  plt.show()
  return(X_train,y_train,X_test)

def main():   
    model = get_model()
    
    # model.load_weights('saved_models/SGD.h5')
    model.load_weights('saved_models/Adam.h5')
    model.save_weights('my_model.h5')

    #live webcam feature detection
    live_AllPoints(model)

    #plot detected points on image with number
    # plot_points('image/image.png', model)

    #detect features on image
    # deal_with_image('images/image.png',model)

    #visualize dataset
    # get_data()

if __name__ == "__main__":
    main()