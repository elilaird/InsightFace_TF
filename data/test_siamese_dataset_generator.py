import tensorflow as tf
import os
from pprint import pprint
from data.mx2tfrecords import parse_function
import argparse

from siamese_dataset import SiameseDatasetGenerator

tf.compat.v1.enable_eager_execution()

def get_parser():
    parser = argparse.ArgumentParser(description='Test dataset generator')
    parser.add_argument('--tfrecords_file_path1', default='../datasets/tfrecords/ms-celeb/tran.tfrecords',help='the image size')
    parser.add_argument('--tfrecords_file_path2', default='../datasets/tfrecords/rally/bh_anchor.tfrecords', help='the image size')
    parser.add_argument('--db_base_path', default='../datasets/faces_ms1m_112x112', help='the image size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('TF Version: ', tf.__version__)
    print('TESTING SIAMESE DATASET GENERATOR')

    args = get_parser()

    file1 = args.tfrecords_file_path
    file2 = args.tfrecords_file_path


    gen = SiameseDatasetGenerator(
            file1,
            file2,
            shuffle=(False,False),
    )

    # bh_dataset = tf.data.TFRecordDataset(file1)
    # bh_dataset = bh_dataset.map(parse_function)
    #bh_dataset = bh_dataset.shuffle(buffer_size=args.buffer_size)
    #bh_dataset = bh_dataset.batch(364)
    #bh_iterator = bh_dataset.make_initializable_iterator()
    #bh_next_element = bh_iterator.get_next()


