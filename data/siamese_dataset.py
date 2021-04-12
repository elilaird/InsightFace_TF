import tensorflow as tf
from data.mx2tfrecords import parse_function
import os



class SiameseDatasetGenerator(object):
    def __init__(self, tfrecord_filepath1, tfrecord_filepath2, shuffle=(False,False), buffer_size=(10000, None), batch_size=(32, 364)):
        tfrecord1 = tfrecord_filepath1
        tfrecord2 = tfrecord_filepath2

        self.dataset1 = tf.data.TFRecordDataset(tfrecord1)
        self.dataset2 = tf.data.TFRecordDataset(tfrecord2)

        # Create a description of the features for the anchor dataset.
        self.feature_description2 = {
            'subjectId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'imageId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'demId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        self.dataset1 = self.dataset1.map(parse_function)
        self.dataset2 = self.dataset2.map(self._parse_function)

        shuffle1, shuffle2 = shuffle
        buffer1, buffer2 = buffer_size
        batch1, batch2 = batch_size

        self.dataset1 = self.dataset1.batch(batch1)
        self.dataset2 = self.dataset2.batch(batch2)

        if shuffle1:
            if buffer1 is not None:
                self.dataset1 = self.dataset1.shuffle(buffer_size=buffer1)
        if shuffle2:
            if buffer2 is not None:
                self.dataset2 = self.dataset2.shuffle(buffer_size=buffer2)



    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.feature_description2)


    def get_iterator(self):
        self.iterator1 = tf.compat.v1.data.make_initializable_iterator(self.dataset1)
        self.iterator2 = tf.compat.v1.data.make_initializable_iterator(self.dataset2)

        ''' out1 = self.iterator1.get_next()
                out2 = self.iterator2.get_next()'''

        return self.iterator1,self.iterator2