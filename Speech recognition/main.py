import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from IPython import display
import matplotlib.pyplot as plt
# from knockknock import sms_sender

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tf_seq2seq_losses.classic_ctc_loss import classic_ctc_loss
from tf_seq2seq_losses.simplified_ctc_loss import simplified_ctc_loss
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def read_and_load_tfrec(tfrec_path):
    raw_tf = tf.io.matching_files(tfrec_path)
    files = tf.random.shuffle(raw_tf)
    shards = tf.data.Dataset.from_tensor_slices(files)
    dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.shuffle(buffer_size=10)
    return dataset

train_dataset = read_and_load_tfrec(tfrec_path = 'inputs/test_tfrecord*')

feature_description = {
    'spectrogram': tf.io.FixedLenFeature([], tf.string),
    'transcription': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    #Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    spectrogram = tf.io.decode_raw(example['spectrogram'],  tf.float32)    
    
    
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(spectrogram, frame_length=255, frame_step=128)
    
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # spectrogram = spectrogram[..., tf.newaxis]
    
    transcription = example['transcription']
    transcription = tf.strings.lower(transcription)
    transcription = tf.strings.unicode_split(transcription, input_encoding="UTF-8")
    transcription = char_to_num(transcription)
    
    return spectrogram, transcription


train_dataset = train_dataset.map(_parse_function, num_parallel_calls = tf.data.AUTOTUNE)

batch_size = 2
# Define the trainig dataset
train_dataset = (
    train_dataset.padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

train_dataset = train_dataset.take(1) #5 training samples.

for element in train_dataset:
    continue

def ctc_loss(y_true, y_pred):
    label_length = tf.reduce_sum(tf.cast(tf.math.not_equal(y_true, 0), tf.int32), axis=1)
    logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  # Assume logit length is the same as y_pred length


    loss = simplified_ctc_loss(
        labels = tf.cast(y_true, dtype = tf.int32),
        logits = y_pred,
        label_length = label_length,
        logit_length = logit_length,
        blank_index = 0
    )
    loss = tf.reduce_mean(loss)
    return loss


def build_model(input_dim, output_dim, rnn_layers=4, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input(shape = (None, input_dim, 1), name = "input", dtype = "float32")
  
    #Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)

    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.LSTM(
            units=rnn_units,
            activation="tanh",
            use_bias=True,
            return_sequences=True
        )
        x = layers.Bidirectional(recurrent, name=f"bidirectional_lstm_{i}",)(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)

     # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax", dtype = "float32")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")

    #Compile model.
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss = ctc_loss, optimizer = optimizer)
    return model


fft_length = 258
print("Building and compiling model.")
model = build_model(
        input_dim=129,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=2,
        rnn_layers = 1
    )


model.fit(train_dataset, epochs = 2, verbose = 1)
