# dialouge manager
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import pairwise_distances_argmin
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *
#Parameters
maxlen = 20
padding = 'pre'
truncating = 'pre'


class ThreadRanker(object):

    def __load_embeddings_by_tag(self, tag_name):
        post_embeddings_file = 'data/title_emb_' + str(tag_name) + ".pickle"
        post_id_file = 'data/post_id_' + str(tag_name) + ".pickle"
        thread_embeddings = unpickle_file(post_embeddings_file)
        thread_ids = unpickle_file(post_id_file)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        prepared_question = question
        question_vec = question_to_vec(prepared_question)
        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings)[0]
        return thread_ids[best_thread]

class DialogueManager(object):
    def __init__(self):
        print("Loading resources...")

        # Intent recognition:
        self.intent_tokenizer = unpickle_file('model/tokenizer_intent.pickle')
        self.intent_recognizer = load_keras_model('model/intent_classifier.h5')
        #self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_tokenizer = unpickle_file('model/tokenizer_tag.pickle')
        self.tag_classifier = load_keras_model('model/tag_classifier.h5')
        self.label_dict = unpickle_file('model/label_dict.pickle')
        self.dict_label = dict((index, key) for key, index in self.label_dict.items())
        self.thread_ranker = ThreadRanker()


    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        prepared_question = text_prepare(question)
        features = self.intent_tokenizer.texts_to_sequences([prepared_question])
        padded = pad_sequences(features, maxlen=maxlen, padding=padding, truncating=truncating)
        intent = self.intent_recognizer.predict(padded)
        intent = intent[0][0]
        if intent < 0.5:
            intent = 'dialogue'
        else:
            intent = 'goal'

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            response = 'Please ask me programming related question only!!'
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            features = self.tag_tokenizer.texts_to_sequences([prepared_question])
            padded = pad_sequences(features, maxlen=maxlen, padding=padding, truncating=truncating)
            tag = np.argmax(self.tag_classifier.predict(padded)[0], axis=-1)

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(features, self.dict_label[tag])

            return self.ANSWER_TEMPLATE % (self.dict_label[tag], thread_id)
