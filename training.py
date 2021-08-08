import os
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import cv2
import tensorflow.keras as keras
import datetime
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.keras.utils.generic_utils import default
from tqdm import tqdm
import time
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Dense
from keras.models import Model, load_model
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import argparse
from tensorflow.keras import mixed_precision
import csv
import math
import seaborn as sns
import random

from vit_keras import vit, utils, visualize, layers
import coral_ordinal as coral
from keras_squeeze_excite_network.se_resnet import SEResNet

# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

def init_parameter():   
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--dataset_path", type=str, default='tfrecords', help="Path della cartella contenente i file tfrecords")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoint', help="Path della cartella in cui salvare i checkpoint del modello")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=10, help="Dimensione dei batch da caricare nella memoria RAM della GPU")
    parser.add_argument("--resume_training", type=int, default=0, help="Impostare a 1 per riprendere un training iniziato precedentemente.")
    parser.add_argument("--model_path", type=str, default='model.h5', help="Path del modello da caricare nel caso in cui resume_training sia posto a 1.")
    parser.add_argument("--image_size", type=int, default=224, help="Dimensione delle immagini.")
    parser.add_argument('--validation', type=int, choices=[0,1], default=0)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--attention_path', type=str,  default='.\\attention_map')
    parser.add_argument('--img_test', type=str,  default='D:\\Tesi\\Dataset\\pre_wiki_2\\25\\16650.jpg')
    parser.add_argument('--ordinal_ranking', type=int, choices=[0,1], default= 1)
    parser.add_argument('--architecture', type=str, choices=['ViT','SEResNet'], default='ViT')
    
    args = parser.parse_args()
    return args
    
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
AUTO = tf.data.experimental.AUTOTUNE

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    return image

def _read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "age": tf.io.FixedLenFeature([], tf.float32),  
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    age = tf.cast(example['age'], tf.float32)
    return image, age

def build_dataset(file_names, batch_size = 19, repeat = 10, one_hot_labels = False):
    #assert(batch_size % 19 == 0)
    return tf.data.TFRecordDataset(file_names).map(
            map_func=_read_labeled_tfrecord,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).repeat(repeat).batch(
            batch_size=batch_size
        ).map(
            map_func = lambda x, y: (vit.preprocess_inputs(tf.cast(x[:,:,:,::-1], tf.float32)), 
                                     tf.one_hot(tf.cast(y, tf.uint8), depth = 100) if one_hot_labels else y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
         ).prefetch(buffer_size = 1000)

def get_VIT(image_size=224, ordinal_ranking = True):
    model = vit.vit_b16(
        image_size=image_size,
        pretrained=True,
        include_top=False,
        pretrained_top=None
    )
    x = coral.CoralOrdinal(num_classes = 100)(model.output) if ordinal_ranking else Dense(100, activation='softmax')(model.output)
    model = Model(model.input, x)
    model.summary()
    return model

def get_SEResNet(image_size = 224, ordinal_ranking = True, weights="imagenet"):
    m1 = SEResNet(weights=weights, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg',weight_decay=0)
    x = coral.CoralOrdinal(num_classes = 100)(m1.output) if ordinal_ranking else Dense(100, activation='softmax')(m1.output)
    model = keras.models.Model(m1.input, x)
    model.summary()
    return model

def write_update(data):
    with open('training.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data) 

def validation(model, valid_set, batch_size, ordinal_ranking = True):
    errors = [[] for _ in range(20)]
    for batch in tqdm(valid_set):
        ages = batch[1]
        pred = model.predict(batch[0])
        pred = np.argmax(pred, axis=1)
        for i, err in enumerate(abs(ages - pred)):
            errors[int(ages[i] // 5)].append(err)
    mae = {'{}-{}'.format(i*5, (i+1)*5) : np.mean(err) for i, err in enumerate(errors) if len(err) > 0}
    print('############### EVALUATION RESULTS ######################')
    for age_range, mae_age in mae.items(): print('MAE {}: {}'.format(age_range, mae_age))
    print('#####################################')
    print('MAE: {}'.format(np.mean(list(mae.values()))))
    print('STD: {}'.format(np.std(list(mae.values()))))
    print('#####################################')


def my_attention_map(model, image):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    size = model.input_shape[1]
    grid_size = int(np.sqrt(model.layers[5].output_shape[0][-2] - 1))

    # Prepare the input
    X = vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]  # type: ignore

    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(
        (num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    fig = plt.figure()
    heat_map = sns.heatmap(mask.reshape(mask.shape[0], mask.shape[1]), 
                            cmap=sns.color_palette(palette="blend:#006400,#008000,#90ee90,#ffff00,#ff8c00,#ff0000,#8b0000", as_cmap=True),
                            xticklabels=False,
                            yticklabels=False,
                            alpha = 0.5)

    heat_map.imshow(image,
            aspect = heat_map.get_aspect(),
            extent = heat_map.get_xlim() + heat_map.get_ylim(),
            )
    plt.tight_layout()

    return fig


class GenerateAttentionMap(keras.callbacks.Callback):

    def __init__(self, output_dir, steps, image_test):
        self.output_dir = output_dir
        self.steps = steps
        self.image_test = image_test
        super().__init__()

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.steps == 0:
            try:
                attention_map = my_attention_map(self.model, self.image_test)
                attention_map.savefig(os.path.join(self.output_dir, "attention_map_epoch-{}_step-{}.png".format(self.epoch, batch)))
                plt.close('all')
                write_update([logs['loss'], logs['mae'] if 'mae' in logs else logs['MeanAbsoluteErrorLabels']])
            except Exception as ex:
                print(ex)

class ValidateModelCallback(keras.callbacks.Callback):

    def __init__(self, validation_set_path, ordinal_ranking, epochs, batch_size):
        self.ordinal_ranking = ordinal_ranking
        self.valid_set = build_dataset(validation_set_path, batch_size=batch_size, repeat = 1)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        validation(self.model,self.valid_set.as_numpy_iterator(), self.ordinal_ranking)

def mae(y_pred, y_true):
    y_true = tf.math.argmax(y_true, axis=1)
    y_pred = tf.math.argmax(y_pred, axis=1)
    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred))

def train(args):
    get_model = get_VIT  if args.architecture == 'ViT' else get_SEResNet
    
    if args.resume_training:
        model = get_model(args.image_size, bool(args.ordinal_ranking))
        model.load_weights(args.model_path)

    else:
        model = get_model(args.image_size, bool(args.ordinal_ranking))
        with open('training.csv', 'w') as f: 
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['loss', 'mae'])


    filepath = 'model-epoch_{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(os.path.join(args.checkpoint_path, filepath), save_weights_only=True, monitor='MeanAbsoluteErrorLabels' if bool(args.ordinal_ranking) else 'mae', verbose=0,save_best_only=True, mode='min', save_freq=5000)
    if args.architecture == 'ViT':
        attention_callback = GenerateAttentionMap(args.attention_path, 3000, cv2.imread(args.img_test, 3)[:,:,::-1])
    validation_callback = ValidateModelCallback(os.path.join(args.dataset_path, 'validation_set', 'validation_set_smaller.tfrecords'), bool(args.ordinal_ranking), args.epochs, args.batch_size)
    if bool(args.ordinal_ranking):
        model.compile(loss = coral.OrdinalCrossEntropy(num_classes = 100), optimizer =  'adam', metrics=[{'MAE' : coral.MeanAbsoluteErrorLabels}]) 
    else:
        model.compile(loss = 'categorical_crossentropy', optimizer =  'adam', metrics = [{'MAE' : mae}])
    train_size = 0
    tf_records = [file for file in os.listdir(args.dataset_path) if '.tfrecords' in file]
    random.shuffle(tf_records)
    
    for i in range(len(tf_records)):
        with open(os.path.join(args.dataset_path, "dataset_{}_size".format(i))) as f:
            train_size += int(f.read())
    
    print('The dataset is composed by {} files for a total of {} images.'.format(len(tf_records), train_size))
    
    callbacks = [checkpoint, validation_callback]
    if args.architecture == 'ViT':
        callbacks.append(attention_callback)

    train_dataset = build_dataset([os.path.join(args.dataset_path, file) for file in tf_records], batch_size = args.batch_size, repeat = args.epochs, one_hot_labels=not bool(args.ordinal_ranking))
    hist = model.fit(train_dataset,
                     epochs = args.epochs,
                     callbacks = callbacks,
                     shuffle = False,
                     initial_epoch = args.last_epoch,
                     steps_per_epoch=math.ceil(train_size / args.batch_size),
                      )
    
    model.save_weights(os.path.join(args.checkpoint_path, "model_epoch-{}.hdf5".format(args.epochs)))
    
    write_update([hist.history['loss'],hist.history['MeanAbsoluteErrorLabels']])

    # train_dataset = train_dataset.as_numpy_iterator()
    # valid_path= os.path.join(args.dataset_path, 'validation_set', 'validation_set_smaller.tfrecords')
    # valid_set = build_dataset(valid_path, batch_size=args.batch_size, repeat = 1)
    # img_test = cv2.imread(args.img_test, 3)[:,:,::-1]
    # for e in range(args.epochs):
    #     print('Epoch {}'.format(e+1))
    #     with tqdm(total = train_size) as pbar:
    #         loss = []
    #         mae = []
    #         for i in range(train_size):
    #             batch = next(train_dataset)
    #             tmp = model.train_on_batch(x = batch[0], y = batch[1])
    #             pbar.set_postfix({'loss': tmp[0], 'mae' : tmp[1]})
    #             pbar.update(1)
    #             loss.append(tmp[0])
    #             mae.append(tmp[1])
    #             if i % 1000 == 0:       
    #                 attention_map = my_attention_map(model, img_test)
    #                 attention_map.savefig(os.path.join(args.attention_path, "attention_map_epoch-{}_step-{}.png".format(e, i)))
    #                 plt.close('all')
    #                 write_update([loss, mae])
    #                 loss = []
    #                 mae = []
    #             if i % 5000 == 0:
    #                 model.save_weights(os.path.join(args.checkpoint_path, 'model-epoch_{}.hdf5'.format(e)))
    #         validation(model,valid_set.as_numpy_iterator(), bool(args.ordinal_ranking))

if __name__ == "__main__":
    args = init_parameter()
    if bool(args.validation):
        get_model = get_VIT
        model = get_model(args.image_size, bool(args.ordinal_ranking))
        model.load_weights(args.model_path, bool(args.ordinal_ranking))
        if bool(args.ordinal_ranking):
            model.compile(loss = coral.OrdinalCrossEntropy(num_classes = 100), optimizer =  'adam', metrics=[{'MAE' : coral.MeanAbsoluteErrorLabels}]) 
        else:
            model.compile(loss = 'categorical_crossentropy', optimizer =  'adam', metrics=[{'MAE' : mae}])
        valid_set = build_dataset(os.path.join(args.dataset_path, 'validation_set', 'validation_set_smaller.tfrecords'), batch_size=batch_size, repeat = args.epochs, one_hot_labels=not bool(args.ordinal_ranking))
        valid_iterator = valid_set.as_numpy_iterator()
        validation(model, valid_iterator, args.batch_size, args.ordinal_ranking)
    else:    
        train(args)
