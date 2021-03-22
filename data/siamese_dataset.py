import tensorflow as tf
from data.mx2tfrecords import parse_function
import os

class SiameseDatasetGenerator(object):
    def __init__(self, tfrecord_filepath1, tfrecord_filepath2, shuffle=(False,False), buffer_size=(10000, None), batch_size=(32, 364)):
        tfrecord1 = os.path.join(tfrecord_filepath1, 'tran.tfrecords')
        tfrecord2 = os.path.join(tfrecord_filepath2, 'tran.tfrecords')

        self.dataset1 = tf.data.TFRecordDataset(tfrecord1)
        self.dataset2 = tf.data.TFRecordDataset(tfrecord2)

        shuffle1, shuffle2 = shuffle
        buffer1, buffer2 = buffer_size
        batch1, batch2 = batch_size

        if shuffle1:
            if buffer1 is not None:
                self.dataset1 = self.dataset1.shuffle(buffer_size=buffer1)
        if shuffle2:
            if buffer2 is not None:
                self.dataset2 = self.dataset2.shuffle(buffer_size=buffer2)

        self.dataset1 = self.dataset1.batch(batch1)
        self.dataset2 = self.dataset2.batch(batch2)

        self.iterator1 = self.dataset1.make_initializable_iterator()
        self.iterator2 = self.dataset2.make_initializable_iterator()


        
    def next(self):
        yield (self.iterator1.get_next(), self.iterator2.get_next())