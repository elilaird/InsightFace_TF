import os
import requests
import tensorflow as tf
import pandas as pd
from pprint import pprint
import argparse
from PIL import Image
from io import BytesIO
import IPython.display as display


print('Tf version: ', tf.__version__)
tf.compat.v1.enable_eager_execution()

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='creates tfrecords from csv file'
    )
    parser.add_argument('--filepath', type=str, required=False, help='path to csv data file')
    parser.add_argument('--tfrecords_file_path', type=str, required=True, help='tfrecords output filename')
    parser.add_argument('--tests', type=bool, default=False, help='test the already created record')
    args = parser.parse_args()
    return args

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example. from: https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(image_string, subjectId, imageId, demId):
    #image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'subjectId': _int64_feature(subjectId),
        'imageId'  : _int64_feature(imageId),
        'demId'    : _int64_feature(demId),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def writeToTFRecord(df, tf_file):
    with tf.io.TFRecordWriter(tf_file) as writer:
        for idx, row in df.iterrows():
            imageId = row['imageId']
            subjectId = row['subject_enc']
            demId = row['dem_enc']
            url = row['url']

            try:
                img_str = requests.get(url).content
            except Exception as e:
                print('Failed to load image: ', url)
                img_str = None

            if img_str is not None:
                tf_example = image_example(img_str, subjectId, imageId, demId)
                writer.write(tf_example.SerializeToString())
            print(idx)

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)



if __name__ == '__main__':
    args = parse_args()

    csv_file = args.filepath
    tfrecords_path = args.tfrecords_file_path

    if not args.tests:
        print('Creating tfrecords file ', tfrecords_path)
        #read csv data
        df = pd.read_csv(csv_file, index_col=0)

        df['dem'] = df['dem'].astype('category')
        df['subject'] = df['subject'].astype('category')
        df['dem_enc'] = df['dem'].cat.codes
        df['subject_enc'] = df['subject'].cat.codes

        writeToTFRecord(df, tfrecords_path)
    else:
        print('Test reading ', tfrecords_path)
        data = tf.data.TFRecordDataset(tfrecords_path)

        # Create a description of the features.
        feature_description = {
            'subjectId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'imageId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'demId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        data = data.map(_parse_function)
        pprint(data)

        for image_features in data.take(1):
            image_raw = image_features['image_raw'].numpy()
            img = Image.open(BytesIO(image_raw))
            img.show()
