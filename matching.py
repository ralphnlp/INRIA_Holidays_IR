import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm
import streamlit as st
from PIL import Image


sift =cv2.xfeatures2d.SIFT_create()
BF = cv2.BFMatcher(cv2.NORM_L2)

def load_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    w, h = 500, 500
    center = [img.shape[0]/2, img.shape[1]/2]
    x = center[1] - w/2
    y = center[0] - h/2
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    #img = cv2.resize(img, (400, 400))
    return crop_img


def img2descriptors(img):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def is_match(descriptors_template, descriptors_source, min_match=30):
    matches = BF.knnMatch(descriptors_template, descriptors_source, k=2)
    good = []
    match_flag = False
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            good.append(m)

    if len(good) >= min_match:
        match_flag = True
    return match_flag, len(good)


def matching(source_img):
    _, descriptors_source = img2descriptors(source_img)
    template_names_in_img, no_goods = [], []
    for i in tqdm(range(len(img_paths))):
        match_flag, no_good = is_match(descriptors_imgs[i], descriptors_source)
        if match_flag == True:
            template_names_in_img.append(img_paths[i])
            no_goods.append(no_good)
    return template_names_in_img, no_goods


def load_database(database_path):
    if os.path.exists(database_path):
        with open(database_path, 'rb') as file:
            descriptors_imgs, no_descriptors_templates, img_paths = pickle.load(file)
    else:
        # Extracting Features of Templates
        descriptors_imgs, no_descriptors_templates, img_paths = [], [], []
        img_dir = './imgs'
        for img_name in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            img = load_img(img_path)
            
            _, descriptors_img= img2descriptors(img)
            if descriptors_img is not None:
                descriptors_imgs.append(descriptors_img)
                no_descriptors_templates.append(descriptors_img.shape[0])
                img_paths.append(img_path)
        with open(database_path, 'wb') as file:
            pickle.dump([descriptors_imgs, no_descriptors_templates, img_paths], file)
    return descriptors_imgs, no_descriptors_templates, img_paths


if __name__=='__main__':

    descriptors_imgs, no_descriptors_templates, img_paths = load_database('./database_sift.pkl')
    file = st.file_uploader(label='')

    if file != None:    
        img = Image.open(file)
        st.image(img, caption='QUERY', width=233)
        img.save('./temp.jpg')

        img_query_path =  './temp.jpg'
        query_img = load_img(img_query_path)
        template_names_in_img, no_goods = matching(query_img)
        no_goods = np.asarray(no_goods)
        indexs = np.flip(no_goods.argsort())[:6]

        imgs = []
        for index in indexs:        
            img = cv2.imread(template_names_in_img[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        st.image(imgs, width=233)
        os.system('rm -rf ./temp.jpg')