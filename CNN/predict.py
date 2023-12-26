import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from modules.experiment_helper import parse_exp_settings
from modules.training_helper import create_model_by_exp_settings
import numpy as np
from pathlib import Path
#from tqdm import tqdm
import tensorflow as tf


def get_test_TFRecord(data_list, batch_size=10000):
    def _parse_example(example_string):
        features_description  = {
            'H8': tf.io.FixedLenFeature([], tf.string),
            'csr': tf.io.FixedLenFeature([], tf.string),
            #'ssi': tf.io.FixedLenFeature([], tf.float32),
            'attr'  : tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, features_description)
        
        H8  = tf.reshape(tf.io.decode_raw(features['H8'], tf.float32), [8,8,1])
        csr = tf.reshape(tf.io.decode_raw(features['csr'], tf.float32),  [8,8,1])
        images = tf.concat([csr, H8], axis=2)
        #images = tf.image.central_crop(H8, 0.5)
        attr = tf.io.decode_raw(features['attr'], tf.float32)
        return images, attr 

    dataset = tf.data.TFRecordDataset(data_list).map(_parse_example)
    return dataset.batch(batch_size)

def main(exp_path, sub_exp_list):
    for sub_exp_name in sub_exp_list:
        # get subexp settings
        sub_exp_settings = parse_exp_settings(exp_path, sub_exp_name)[0]
        
        # load model
        exp_name = sub_exp_settings['experiment_name']
        model_save_path = f'saved_models/{exp_name}/{sub_exp_name}'
        model = create_model_by_exp_settings(sub_exp_settings['model'], sub_exp_settings['model_setting'], model_save_path)
        print('Start predict sub-exp:', sub_exp_name)
        
        for yr in [2022,2023]:
            dataset = get_test_TFRecord(f'../data/2TFR/H8csr_s8_{yr}.tfr')
            # output 
            predicts = []
            for image, attr in dataset:
                pred =  np.squeeze(model(image, attr))
                predicts.extend(pred)

            save_folder = f'../output/{exp_name}'
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            np.save(f'{save_folder}/{sub_exp_name}{yr}.npy', predicts)
    print('All prediction done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path')
    parser.add_argument('-s', '--sub_exp_list', nargs='+')
    args = parser.parse_args()

    main(args.exp_path, args.sub_exp_list)
