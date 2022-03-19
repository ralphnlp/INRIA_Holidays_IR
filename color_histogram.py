import os
from cv2 import cvtColor
from matplotlib.pyplot import hsv
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.neighbors import KDTree
import time
import pickle
import streamlit as st
from PIL import Image


def load_img(img_path, shape = (400, 400), gray = False, hsv = False):
    h ,w = shape
    img = cv2.imread(img_path)
    img = cv2.resize(img, (h, w))

    if hsv == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if gray == True:
        img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.reshape(1, h, w)
    return img.reshape(3, h, w)


def img2ColorHis(img):
    B, G, R = np.zeros((3, 128))
    def transform(his, matrix):
        matrix = matrix.reshape(-1)
        for i in matrix:
            his[i//2] += 1
        return his
    B, G, R = transform(B, img[0]), transform(G, img[1]), transform(R, img[2])
    feature_vector = np.concatenate([B, G, R])
    return feature_vector
    

def img2GrayHis(img):
    gray = img.reshape(-1)
    Gr = np.zeros((1, 128))
    for value in gray:
        Gr[0][value//2] += 1
    return Gr


def load_database(database_path):
    if os.path.exists(database_path):
        print('[INFO] Exists database...')
        with open(database_path, 'rb') as file:
            features, img_paths = pickle.load(file)
    else:
        features = np.zeros((1, 384))
        img_paths = []
        img_dir = './imgs'
        for img_name in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            img = load_img(img_path, hsv=True)
            feature = img2ColorHis(img)
            img_paths.append(img_path)
            features = np.concatenate([features, feature.reshape(1, -1)], axis=0)
        features = features[1:]
        print(f'the feature shape = {features.shape}')
        with open(database_path, 'wb') as file:
            pickle.dump([features, img_paths], file)
    return features, img_paths


def build_kdtree(features):
    s_time = time.time()
    kdtree = KDTree(features, leaf_size = 20)
    print(f'[INFO] The time for building kdtree = {round(time.time()-s_time, 2)}s')
    return kdtree


if __name__=='__main__':

    features, img_paths = load_database(database_path='./database_hsv.pkl')
    kdtree = build_kdtree(features)
    print('[INFO] Query...')

    file = st.file_uploader(label='')
    if file != None:
        
        imgs = []
        img = Image.open(file)
        imgs.append(img)
        img.save('./temp.jpg')

        img = load_img(img_path='./temp.jpg', hsv=True)
        feature = img2ColorHis(img)
        
        _, indexs = kdtree.query(feature.reshape(1, -1), k=5)
        indexs = indexs[0]
        
        for index in indexs:
            img = cv2.imread(img_paths[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        
        st.image(imgs, width=233)
        os.system('rm -rf ./temp.jpg')