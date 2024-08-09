import importlib 
import tensorflow as tf


def set_up_tensorflow(GPU_limit):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')
    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )
    
    
def get_summary_writer(log_path):
    return tf.summary.create_file_writer(log_path)


def get_TFRecord_dataset(data_yr, shuffle_buffer, batch_size):
    # load data
    print('Data loading...')
    def _parse_example(example_string):
        features_description  = {
            'H8': tf.io.FixedLenFeature([], tf.string),
            'csr' : tf.io.FixedLenFeature([], tf.string),
            'ssi': tf.io.FixedLenFeature([], tf.float32),
            'attr': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, features_description)
        
        H8 = tf.reshape(tf.io.decode_raw(features['H8'], tf.float32), [8,8,1])
        csr = tf.reshape(tf.io.decode_raw(features['csr'], tf.float32), [8,8,1])
        dcsr = tf.subtract(csr, H8)
        images = tf.concat([H8, dcsr], axis=2)
        attr = tf.io.decode_raw(features['attr'], tf.float32)
        return images, attr, features['ssi']
    
    data_file = [f'../data/2TFR_v3/H8csr_s8_{yr}.tfr' for yr in range(data_yr[0], data_yr[1]+1)] #!!!
    dataset = tf.data.TFRecordDataset(data_file).map(_parse_example)
    
    # split to train/valid
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    count = sum(1 for _ in dataset)
    train_size = int(count*0.7)
    ds_for_model ={
        'train' : dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE),
        'valid' : dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    }
    return ds_for_model


def create_model_by_exp_settings(model_name, model_setting, load_from=''):
    tf.keras.backend.clear_session()
    print('Create model...')
    model_class = importlib.import_module(f'models.{model_name}').Model(**model_setting)
    if load_from:
        model_class.load_weights(load_from+'/M').expect_partial()
    return model_class
