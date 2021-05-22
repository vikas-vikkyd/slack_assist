import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import pickle
import tensorflow as tf
import numpy as np

def text_prepare(text):
    """Preform tokenization simple preprocessing"""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    return str(text.strip())

def unpickle_file(path):
    with open(path, 'rb') as handle:
        pickle_file = pickle.load(handle)
    return pickle_file

def load_keras_model(path):
    return tf.keras.models.load_model(path)
def question_to_vec(question):
    tag_classifier = load_keras_model('model/tag_classifier.h5')
    word_embeddings = tag_classifier.layers[0].get_weights()[0]
    i = 0
    vec = np.zeros(50)
    for idx in question[0]:
        vec = vec + word_embeddings[idx]
        i = i + 1
    vec_arr = vec/i
    vec_arr = vec_arr.reshape(1, -1)
    return vec_arr