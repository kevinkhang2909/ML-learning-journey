import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

NUM_SHARDS = 16
IMAGE_SIZE = (512, 512)
SEED = 42


def encode_image(filepath, method='bilinear'):
    image_string = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Must convert dtype to float32 for most resizing methods to work
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method=method, antialias=True)

    # Convert dtype to uint8 to be encoded to bytestring for tfrec
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True)
    return image


def featurize(val):
    if isinstance(val, (bytes, str, tf.Tensor)):
        if isinstance(val, type(tf.constant(0))):
            val = val.numpy()
        elif isinstance(val, str):
            val = str.encode(val)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
    elif isinstance(val, (int, np.integer)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
    elif isinstance(val, (float, np.floating)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))
    else:
        raise Exception(f'Cannot featurize due to type {type(val)}')


def serialize_example(row):
    feature = row.to_dict()
    img_path = os.path.join(path_img, feature['image'])
    feature['image'] = encode_image(img_path)
    feature['matches'] = tf.io.serialize_tensor(tf.convert_to_tensor(feature['matches']))
    for k, v in feature.items():
        feature[k] = featurize(v)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfr(df, filepath, filename, file_index, file_size, image_indexes):
    with tf.io.TFRecordWriter(str(filepath / f'{filename}{file_index}.tfrec')) as writer:
        start = file_size * file_index
        end = file_size * (file_index + 1)
        for i in tqdm(image_indexes[start:end]):
            example = serialize_example(df.loc[i])
            writer.write(example)


path = Path.home() / 'OneDrive - Seagroup/computer_vison/shopee_item_images/'
path_img = path / 'train_images'
df = pd.read_csv(path / 'train.csv')

match_map = df.groupby(['label_group'])['posting_id'].unique().to_dict()
df['matches'] = df['label_group'].map(match_map)

label_mapper = dict(zip(df['label_group'].unique(), np.arange(len(df['label_group'].unique()))))
df['label_group'] = df['label_group'].map(label_mapper)

path_record = path / 'train_record'
image_indexes = df.index.values
file_size = len(image_indexes) // 15
file_count = len(image_indexes) // file_size + int(len(image_indexes) % file_size != 0)
for file_index in range(file_count):
    print('Writing TFRecord %i of %i...' % (file_index, file_count))
    write_tfr(df, path_record, 'train', file_index, file_size, image_indexes)
