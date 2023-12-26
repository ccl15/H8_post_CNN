import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# read 
def parse_tfrecord(example_proto):
    features = {
        'H8': tf.io.FixedLenFeature([], tf.string),
        'ssi': tf.io.FixedLenFeature([], tf.float32),
        'attr': tf.io.FixedLenFeature([], tf.string),
        'csr': tf.io.FixedLenFeature([], tf.string),
        'time': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features

# write
def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


fin = [f'2TFR/H8csr_s8_{yr}.tfr' for yr in range(2022,2024)]
dataset = tf.data.TFRecordDataset(fin).map(parse_tfrecord)
for hr in [12,17]:
    # Filter records where 'time'
    filtered_dataset = dataset.filter(lambda x: tf.strings.substr(x['time'], -2, 2) == str(hr))

    # Save the filtered dataset to a new TFRecord file
    fout = f'3hrTFR/y2223_h{hr}.tfr'
    with tf.io.TFRecordWriter(fout) as writer:
        for record in filtered_dataset:
            feature = {
                'H8': _bytes_feature(record['H8'].numpy()),
                'ssi': _float_feature(record['ssi'].numpy()),
                'attr': _bytes_feature(record['attr'].numpy()),
                'csr': _bytes_feature(record['csr'].numpy()),
                'time': _bytes_feature(record['time'].numpy()),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
           
#%% check value
'''
dataset = tf.data.TFRecordDataset('2TFR/H8csr_s8_2016.tfr').map(parse_tfrecord)
features = dataset.filter(lambda x: x['time'] == '2016071012')

for feature in features:
    H8a  = tf.reshape(tf.io.decode_raw(feature['H8'], tf.float32), [8,8])
    ssia = feature['ssi']
    break


dataset = tf.data.TFRecordDataset('3hrTFR/y1621_h12.tfr').map(parse_tfrecord)
features = dataset.filter(lambda x: x['time'] == '2016071012')

for feature in features:
    H8b  = tf.reshape(tf.io.decode_raw(feature['H8'], tf.float32), [8,8])
    ssib = feature['ssi']
    break

print(H8a-H8b)
print(ssia-ssib)
'''