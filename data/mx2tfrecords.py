import mxnet as mx
import argparse
import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import base64
import os
from pprint import pprint
#tf.contrib.eager.enable_eager_execution()

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--new', default=False, help='Create new dataset from recordio', type=bool)
    args = parser.parse_args()
    return args


def mx2tfrecords_old(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        encoded_jpg_io = io.BytesIO(img)
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img_raw = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        try:
            if isinstance(header.label,list):
                label = int(header.label[0])
            else:
                label = int(header.label)
        except Exception as e:
            print(type(header.label))
            exit()
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


if __name__ == '__main__':

    args = parse_args()

    if args.new:

        # # define parameters
        idxList = []
        data_shape = (3, 112, 112)

        imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
        s = imgrec.read()
        header, _ = mx.recordio.unpack(s)
        imgrec.reset()
        while s is not None:
            s = imgrec.read()
            if s is not None:
                header, _ = mx.recordio.unpack(s)
                idxList.append(header.id)

        # generate tfrecords
        imgrec.reset()
        mx2tfrecords(idxList, imgrec, args)

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #sess = tf.Session(config=config)
    # training datasets api config
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=30000)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # begin iteration
    '''for i in range(10):
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                print(images.shape)
                break
            except tf.errors.OutOfRangeError:
                print("End of dataset")'''




