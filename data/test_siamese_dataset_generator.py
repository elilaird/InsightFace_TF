import tensorflow as tf
import os
from pprint import pprint
from data.mx2tfrecords import parse_function
import argparse
from PIL import Image
from io import BytesIO

from siamese_dataset import SiameseDatasetGenerator

#tf.compat.v1.enable_eager_execution()

def get_parser():
    parser = argparse.ArgumentParser(description='Test dataset generator')
    parser.add_argument('--tfrecords_file_path1', default='../datasets/ms-celeb/tran.tfrecords',help='the image size')
    parser.add_argument('--tfrecords_file_path2', default='../datasets/rally/bh_anchor.tfrecords', help='the image size')
    parser.add_argument('--db_base_path', default='../datasets/faces_ms1m_112x112', help='the image size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('TF Version: ', tf.__version__)
    print('TESTING SIAMESE DATASET GENERATOR')

    args = get_parser()

    file1 = args.tfrecords_file_path1
    file2 = args.tfrecords_file_path2


    gen = SiameseDatasetGenerator(
            file1,
            file2,
            shuffle=(False,False)
    )

    '''    #eager tests
    d1 = gen.dataset1
    d2 = gen.dataset2

    pprint(d2)

    for image_features in d2.take(1):
        image_raw = image_features['image_raw'].numpy()
        img = Image.open(BytesIO(image_raw))
        img.show()'''



    #define session
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        ms_iter, a_iter = gen.get_iterator()


        sess.run(ms_iter.initializer)
        sess.run(a_iter.initializer)

        ms_images, ms_labels = sess.run(ms_iter.get_next())
        a_dict = sess.run(a_iter.get_next())





        print("MS images shape: ", ms_images.shape)
        print("MS labels", len(ms_labels))
        print("Anchor images shape: ", a_dict.keys())














