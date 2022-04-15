from deep_ir import *
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/query', methods=['POST'])
def query():
    
    path = request.form.get('img_path')
    img = Utils.load_img(path)
    start_time = time.time()

    input_query = feature_extractor.extract(img.reshape(1, *img.shape))
    input_query = siamese_net.predict(input_query)
    indexs, _ = matching_model.query(input_query)

    imgs = []
    for index in indexs:
        imgs.append(img_paths[index])
    print(f'[INFO] Time quering = {time.time()-start_time}') 

    reponse = {'result': imgs}
    return jsonify(reponse)


if __name__ == '__main__':

    # init models
    siamese_net = Siamese_Net()
    feature_extractor = Feature_Extractor_VGG16()

    # load database
    features, img_paths = Utils.load_database('./database_feature_extraction.pkl')

    # generate trainset
    index_dataset = Utils.generate_trainset(img_paths)
    anchor, pos, neg = [], [], []
    for i in tqdm(range(index_dataset.shape[0])):
        a_path, p_path, n_path = index_dataset[i]
        a_i, p_i, n_i = Utils.path2id(a_path, img_paths), Utils.path2id(p_path, img_paths), Utils.path2id(n_path, img_paths)
        anchor.append(features[a_i])
        pos.append(features[p_i])
        neg.append(features[n_i])
    anchor = np.asarray(anchor)
    pos = np.asarray(pos)
    neg = np.asarray(neg)

    # train siamese_net
    siamese_net.fit([anchor, pos, neg], epochs=2, lr=0.1)
    features = siamese_net.predict(features)

    # init matching model
    matching_model = Matching(features)  
    
    app.run(debug=True, host='127.0.0.2', port='2222')
