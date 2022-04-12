
import streamlit as st
from PIL import Image
import requests
import cv2
import os

if __name__=='__main__':


    url = ' http://127.0.0.2:2222/query'
    file = st.file_uploader(label='')
    if file != None:
        img = Image.open(file)
        st.image(img, caption='QUERY', width=233)
        save_path = './temp.jpg'
        img.save(save_path)

        data = {'img_path': save_path}
        print(data)
        reponse = requests.post(url, data=data).json()
        imgs_path, time_query = reponse['result'], reponse['time_query']

        imgs = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        
        imgs = imgs[1:]
        st.write(f'time query = {time_query}')
        st.image(imgs, width=233)
        os.system('rm -rf ./temp.jpg')