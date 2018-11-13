from libact.base.interfaces import Model, ContinuousModel, ProbabilisticModel
import tensorflow as tf
import numpy as np


class MyModelTF(ProbabilisticModel):
    """
    读取Tensorflow模型
    """

    def __init__(self, model_folder, *args, **kwargs):
        sess = tf.Session()
        input_graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(model_folder, 'rb') as f: 
            input_graph_def.ParseFromString(f.read())
        tf.import_graph_def(input_graph_def, name="")
            
        self.input_tensor = sess.graph.get_tensor_by_name('input:0')
        self.output_tensor = sess.graph.get_tensor_by_name('output/sigm:0')
        self.sess = sess
        self.batch_size = kwargs.pop('batch_size', 32)

    def train(self, dataset, *args, **kwargs):
        return 0
    #     x, y = dataset.format_sklearn()
    #     feed_dict = {self.input_tensor: x, self.output_tensor: y}
    #     result_predicted = self.sess.run(self.output_tensor, feed_dict=feed_dict)
    #     return self._model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        
        for i in range(int(np.ceil(len(feature)/ self.batch_size))):
            if (i+1)* self.batch_size > len(feature):
                data = feature[i * self.batch_size : len(feature)]
            else:
                data = feature[i * self.batch_size : (i+1) * self.batch_size]
            feed_dict = {self.input_tensor: data}
            if i == 0:
                result_predicted = self.sess.run(self.output_tensor, feed_dict=feed_dict)
            else:
                temp_predicted = self.sess.run(self.output_tensor, feed_dict=feed_dict)
                result_predicted = np.row_stack((result_predicted, temp_predicted))
        return result_predicted

    def score(self, testing_dataset, *args, **kwargs):
        x, y_ = testing_dataset.format_sklearn()
        
        for i in range(int(np.ceil(len(x)/ self.batch_size))):
            if (i+1)* self.batch_size > len(x):
                data = x[i * self.batch_size : len(x)]
            else:
                data = x[i * self.batch_size : (i+1) * self.batch_size]
            feed_dict = {self.input_tensor: data}
            if i == 0:
                y = self.sess.run(self.output_tensor, feed_dict=feed_dict)
            else:
                temp_y = self.sess.run(self.output_tensor, feed_dict=feed_dict)
                y = np.row_stack((y, temp_y))
        
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = self.sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

        return acc

    def predict_proba(self, feature, *args, **kwargs):
        return self.predict(feature, *args, **kwargs)