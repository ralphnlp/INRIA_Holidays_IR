import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm
import pickle
import numpy as np
import cv2
from sklearn.neighbors import KDTree
import torch
import time


class Feature_Extractor_VGG16:
    
    model = VGG16()
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    for layer in model.layers[:]:
        layer.trainable = False

    def __init__(self) -> None:
        pass

    def extract(self, imgs):
        imgs_shape = imgs.shape
        features = None
        if imgs_shape[1] == 224 and imgs_shape[2] == 224:
            features = self.model.predict(imgs)
        else:
            print('reshape img to (224, 224)')
        return features


class Siamese_Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=4096, out_features=1024)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float)
        outputs = self.linear1(X)
        outputs = self.sigmoid(outputs)
        return outputs


    def _triple_loss(self, a, p, n):
        loss = torch.cosine_similarity(a, n)**2 - torch.cosine_similarity(a, p)
        loss = torch.mean(loss, dim=0)
        return loss


    def fit(self, train_set, epochs, lr, val_set = None):
        
        anchor, pos, neg = train_set
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
        
            a, p ,n = self.forward(anchor), self.forward(pos), self.forward(neg)
            loss = self._triple_loss(a, p, n)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            with torch.no_grad():
                if epoch % 5 == 0:
                    print(f'epoch {epoch}th -> loss = {loss}')
    

    def predict(self, X):
        with torch.no_grad():
            features = self.forward(X)
            return features.cpu().numpy()


class Matching:

    def __init__(self, database) -> None:
        features = database
        s_time = time.time()
        self.kdtree = KDTree(features, leaf_size = 20)
        print(f'[INFO] Build KDTree successfully in {time.time()-s_time}s')

    def query(self, input_query):
        feature = input_query
        s_time = time.time()
        _, indexs = self.kdtree.query(feature.reshape(1, -1), k=7)
        query_time = time.time()-s_time
        print(f'[INFO] Query successfully in {query_time}s')
        return indexs[0], query_time


class Utils:

    def load_img(img_path, shape = (224, 224)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, shape)
        return img


    def load_database(database_path):
        if os.path.exists(database_path):
            print('[INFO] Exists database...')
            with open(database_path, 'rb') as file:
                features, img_paths = pickle.load(file)
        else:

            feature_extractor = Feature_Extractor_VGG16()
            features = np.zeros((1, 4096))
            img_paths = []
            img_dir = './imgs'

            for img_name in tqdm(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, img_name)
                img = Utils.load_img(img_path)
                feature = feature_extractor.extract(img.reshape(1, *img.shape))
                img_paths.append(img_path)
                features = np.concatenate([features, feature.reshape(1, -1)], axis=0)
            features = features[1:]

            print(f'the feature shape = {features.shape}')
            with open(database_path, 'wb') as file:
                pickle.dump([features, img_paths], file)
        return features, img_paths


    def generate_trainset(img_paths):

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
