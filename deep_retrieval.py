import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm
import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify


def load_img(img_path, shape = (224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, shape)
    return img


def init_feature_extractor():
    feature_extractor = VGG16()
    #output = Dense(1024, activation='sigmoid')(feature_extractor.layers[-2].output)
    model = Model(inputs=feature_extractor.input, outputs=feature_extractor.layers[-2].output)

    for layer in model.layers[:]:
        layer.trainable = False
    print(model.summary())
    return model


def img2feature(img):
    feature_vector = feature_extractor.predict(img.reshape(1, 224, 224, 3))[0]
    return feature_vector


def load_database(database_path):
    if os.path.exists(database_path):
        print('[INFO] Exists database...')
        with open(database_path, 'rb') as file:
            features, img_paths = pickle.load(file)
    else:
        features = np.zeros((1, 4096))
        img_paths = []
        img_dir = './imgs'
        for img_name in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            img = load_img(img_path)
            feature = img2feature(img)
            img_paths.append(img_path)
            features = np.concatenate([features, feature.reshape(1, -1)], axis=0)
        features = features[1:]
        print(f'the feature shape = {features.shape}')
        with open(database_path, 'wb') as file:
            pickle.dump([features, img_paths], file)
    return features, img_paths


import torch

class Triple_Model(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=4096, out_features=1024)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        outputs = self.linear1(X)
        outputs = self.sigmoid(outputs)
        return outputs


def triple_loss(a, p, n):
    loss = torch.cosine_similarity(a, n)**2 - torch.cosine_similarity(a, p)
    loss = torch.mean(loss, dim=0)
    return loss
    

def neg_sampling(img_paths):

    img_names = [path.split('/')[-1] for path in img_paths]
    img_names = [name.split('.')[0] for name in img_names]
    
    key2paths = {}
    for i, name in enumerate(img_names):
        if name[:4] in key2paths.keys():
            key2paths[name[:4]] = np.concatenate([key2paths[name[:4]], np.asarray([img_paths[i]])])
        else:
            key2paths[name[:4]] = np.asarray([img_paths[i]])
    

    index_dataset = []
    for key_cls in key2paths.keys():
        a, p, n = [], [], []
        n_sample = 10

        random = np.random.randint(low=0, high=len(key2paths[key_cls]), size=(n_sample))
        a.append(key2paths[key_cls][random])

        random = np.random.randint(low=0, high=len(key2paths[key_cls]), size=(n_sample))
        p.append(key2paths[key_cls][random])

        choices_clss = list(key2paths.keys())
        choices_clss.remove(key_cls)
        random_cls = np.random.randint(low=0, high=len(choices_clss), size=(n_sample))

        for cls in random_cls:
            name_path = np.random.choice(key2paths[choices_clss[cls]], size=1)
            n.append(name_path)

        a = np.asarray(a)
        p = np.asarray(p)
        n = np.asarray(n)
        temp = np.concatenate([a.copy(), p.copy(), n.copy().reshape(1, -1)]).T
        index_dataset.append(temp)
    
    index_dataset = np.concatenate(index_dataset, axis=0)
    print(index_dataset.shape)
    return index_dataset


def path2id(path_query, paths):
    for i, path in enumerate(paths):
        if path == path_query:
            return i



import time

app = Flask(__name__)
@app.route('/query', methods=['POST'])
def query():
    
    path = request.form.get('img_path')
    start_time = time.time()

    img = load_img(path)
    feature = img2feature(img)
    feature = torch.as_tensor(feature, dtype=torch.float).reshape(1, -1)
    with torch.no_grad():
        feature = model(feature)[0].numpy()
    _, indexs = kdtree.query(feature.reshape(1, -1), k=7)
    indexs = indexs[0]
    imgs = []
    for index in indexs:
        imgs.append(img_paths[index])
    reponse = {'result': imgs, 'time_query': time.time()-start_time}
    #print(reponse)
    return jsonify(reponse)


from sklearn.neighbors import KDTree
def build_kdtree(features):
    s_time = time.time()
    kdtree = KDTree(features, leaf_size = 20)
    print(f'[INFO] The time for building kdtree = {round(time.time()-s_time, 2)}s')
    return kdtree


if __name__ == '__main__':

    print(torch.cuda.is_available())
    
    feature_extractor = init_feature_extractor()
    features, img_paths = load_database('./database_feature_extraction.pkl')
    index_dataset = neg_sampling(img_paths)
    archor, pos, neg = [], [], []
    
    for i in tqdm(range(index_dataset.shape[0])):
        a_path, p_path, n_path = index_dataset[i]
        a_i, p_i, n_i = path2id(a_path, img_paths), path2id(p_path, img_paths), path2id(n_path, img_paths)
        archor.append(features[a_i])
        pos.append(features[p_i])
        neg.append(features[n_i])
    
    archor = np.asarray(archor)
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    print(archor.shape, pos.shape, neg.shape)


    epochs = 50
    model = Triple_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    archor = torch.as_tensor(archor, dtype=torch.float)
    pos = torch.as_tensor(pos, dtype=torch.float)
    neg = torch.as_tensor(neg, dtype=torch.float)

    for epoch in range(epochs):

        a, p ,n = model(archor), model(pos), model(neg)
        loss = triple_loss(a, p, n)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            print(f'epoch {epoch}th -> loss = {loss}')
    

    with torch.no_grad():
        private_features = []
        for feature in tqdm(features):
            feature = torch.as_tensor(feature, dtype=torch.float).reshape(1, -1)
            private_feature = model(feature)[0].numpy()
            private_features.append(private_feature)
        
        private_features = np.asarray(private_features)
        print(private_features.shape)
        #with open('./private_features.pkl', 'wb') as file:
        #    pickle.dump([private_features, img_paths], file)

    kdtree = build_kdtree(private_features)
    app.run(debug=True, host='127.0.0.2', port='2222')