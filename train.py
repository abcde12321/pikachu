import numpy as np
import cv2 as cv
import os
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
import pathlib
import pickle

def list_image_file(directory):
  image_list = []
  for f in os.listdir(directory):
    #Todo gif format
    if f.endswith(".jpg") or f.endswith(".png"):
      image_list.append(f)
      #print(f)
  return image_list

def extract_features(list_image_file,path):
  data_features = []
  radius = 1
  n_points = 8 * radius
  image_size = 256

  for filename in list_image_file:
    img = cv.imread(os.path.join(path,filename), cv.IMREAD_GRAYSCALE)
    #print('start to resize image',os.path.join(path,filename))
    img = cv.resize(img, (image_size, image_size))
    lbp_feature = local_binary_pattern(img,n_points,radius)
    data_features.append(lbp_feature.flatten())
  return data_features


def main():
  path_pikachu_images = os.path.join(os.path.dirname(__file__), "pikachu_dataset/pikachu/")
  path_not_pikachu_images = os.path.join(os.path.dirname(__file__),"pikachu_dataset/not_pikachu/")

  list_pikachu_images = list_image_file(path_pikachu_images)
  print("we got", len(list_pikachu_images), "pikachu images")
  list_not_pikachu_images = list_image_file(path_not_pikachu_images)
  print("we got", len(list_not_pikachu_images), "not_pikachu images")

  # use 400 for train, rest for testing, no validation for logist regression
  nr_of_train = 400
  #nr_of_validation = 100

  print('start loading features....')
  features_pikachu = np.asarray(extract_features(list_pikachu_images,path_pikachu_images))
  features_not_pikachu = np.asarray(extract_features(list_not_pikachu_images,path_not_pikachu_images))
  print("pikachu_features:", features_pikachu.shape,", not_pikachu_features:", features_not_pikachu.shape)

  X_train = np.concatenate((features_pikachu[:nr_of_train], features_not_pikachu[:nr_of_train]), axis=0)
  labels_pikachu = np.ones(nr_of_train)
  labels_not_pikachu = np.zeros(nr_of_train)
  Y_train = np.concatenate((labels_pikachu, labels_not_pikachu), axis=0)
  print("X_train:",X_train.shape,", Y_train:",Y_train.shape)

  X_test = np.concatenate((features_pikachu[nr_of_train:len(list_pikachu_images)], features_not_pikachu[nr_of_train:len(list_not_pikachu_images)]), axis=0)
  Y_test = np.concatenate((np.ones(len(list_pikachu_images)-nr_of_train), np.zeros(len(list_not_pikachu_images)-nr_of_train)), axis=0)
  print("X_test:",X_test.shape,", Y_test:",Y_test.shape)
  
  #train model
  logisticRegr = LogisticRegression(C=1,penalty="l1")
  lg_model = logisticRegr.fit(X_train, Y_train)

  #evulate model
  print('train accuracy:',lg_model.score(X_train, Y_train))
  print('test accuracy:',lg_model.score(X_test, Y_test))

  #save model
  path_model = os.path.join(os.path.dirname(__file__), "model/lr.pkl")
  with open(path_model, 'wb') as f:
    pickle.dump(logisticRegr, f) 


main()
print ("finished.")