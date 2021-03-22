import tensorflow as tf
import os
from pprint import pprint
import argparse

from siamese_dataset import SiameseDatasetGenerator

def get_parser():
    parser = argparse.ArgumentParser(description='Test dataset generator')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords/rally', help='the image size')
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

    for _ in range(5):
        val1, val2 = gen.next()
        pprint(val1)
        pprint(val2)

