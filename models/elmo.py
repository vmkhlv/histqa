import tensorflow as tf
import tensorflow_hub as hub
from .utils import singleton


@singleton
class ElmoEmbedder:
    def __init__(self, hub_url="http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz"):
        self.hub_url = hub_url
        self.device = tf.test.gpu_device_name()
        self.elmo_model, self.graph = self.initialize_model()

    def initialize_model(self):
        graph = tf.Graph()
        with graph.as_default():
            elmo_model = hub.Module(self.hub_url, trainable=True)
        return elmo_model, graph

    def preprocess_batch(self, batch):
        max_elem = max([len(batch_elem) for batch_elem in batch])
        batch_elems, batch_lens = [], []
        for batch_elem in batch:
            batch_lens.append(len(batch_elem))
            # padding
            if len(batch_elem) < max_elem:
                batch_elem = batch_elem + [""] * (max_elem - len(batch_elem))
            batch_elems.append(batch_elem)
        return {
            "tokens": batch_elems,
            "sequence_len": batch_lens
        }

    def batch_to_embeddings(self, batch):
        """
        :param batch: list of lists
        :return: elmo embeddings of shape (batch_size, 3, max_length, 1024)
        """
        batch_inputs = self.preprocess_batch(batch)
        with tf.device(self.device):
            with tf.compat.v1.Session(graph=self.graph) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.tables_initializer())
                layers = self.elmo_model(inputs=batch_inputs, signature="tokens", as_dict=True)
                # (batch_size, max_length, 512)
                char_cnn = sess.run(layers["word_emb"])
                # (batch_size, max_length, 1024)
                lstm1 = sess.run(layers["lstm_outputs1"])
                # (batch_size, max_length, 1024)
                lstm2 = sess.run(layers["lstm_outputs2"])
                # (batch_size, max_length, 1024)
                char_cnn = tf.concat([char_cnn, char_cnn], axis=-1)
                # (batch_size, 3, max_length, 1024)
                embeddings = tf.concat(
                    [tf.expand_dims(layer, axis=1) for layer in [char_cnn, lstm1, lstm2]], axis=1
                ).eval()
                return embeddings
