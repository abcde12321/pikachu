import sys, getopt
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv
import pickle
import os
import pathlib

#mostly copy paste from train.py
def extract_features(path_image):
    print('extract feature from image:',path_image)
    radius = 1
    n_points = 8 * radius
    image_size = 256

    img = cv.imread(path_image, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (image_size, image_size))
    lbp_feature = local_binary_pattern(img,n_points,radius)
    return lbp_feature.flatten()

def predict(model, imgs_path):
    model_path = os.path.join(os.path.dirname(__file__), 'model/lr.pkl') #+ model)
    print('load model from', model_path)
    model_load = pickle.load(open(model_path, 'rb'))
    
    path_image = os.path.join(os.path.dirname(__file__), imgs_path)
    print('extra test images in', path_image)
    for img in os.listdir(path_image):
        path = os.path.join(path_image, img) 
        feature = extract_features(path)
        prediction = model_load.predict([feature])
        print(prediction)
        print('predict ',img, 'is' if prediction==1 else 'is not','pikachu' )

def main(argv):
    model = ''
    img_path = ''
    try:
        opts, args = getopt.getopt(argv,"hm:f:",["model=","path="])
    except getopt.GetoptError:
        print ('predict.py -m <model> -f <path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('predict.py -m <model> -f <path>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg +'.pkl'
        elif opt in ("-f", "--path"):
            img_path = arg

    print('use model:',model,'to predict:',img_path)
    predict(model,img_path)


if __name__ == "__main__":
    main(sys.argv[1:])