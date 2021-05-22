from utils import load_keras_model, unpickle_file
import pandas as pd
import numpy as np
import pickle


def create_embeddings():
    tag_tokenizer = unpickle_file('model/tokenizer_tag.pickle')
    tag_classifier = load_keras_model('model/tag_classifier.h5')
    word_embeddings = tag_classifier.layers[0].get_weights()[0]
    data = pd.read_csv('data/tagged_posts.tsv', sep='\t')
    for tag in data['tag'].unique():
        tag_embeddings = []
        post_ids = data[data['tag'] == tag]['post_id'].values
        for index, row in data[data['tag'] == tag].iterrows():
            i = 0
            vec = np.zeros(50)
            for idx in tag_tokenizer.texts_to_sequences([row['title']])[0]:
                vec = vec + word_embeddings[idx]
                i = i + 1
            tag_embeddings.append(vec/i)
        filename_post_id = 'data/post_id_' + tag + '.pickle'
        filename_title_emb = 'data/title_emb_' + tag + '.pickle'
        with open(filename_post_id, 'wb') as handle:
            pickle.dump(post_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(filename_title_emb, 'wb') as handle:
            pickle.dump(tag_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)