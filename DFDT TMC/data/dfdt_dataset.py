'''Module for loading the fakeavceleb dataset from tfrecord format'''
import numpy as np
import tensorflow as tf

FEATURE_DESCRIPTION = {
    'video_path': tf.io.FixedLenFeature([], tf.string), 
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'clip/label/index': tf.io.FixedLenFeature([], tf.int64),
    'clip/label/text': tf.io.FixedLenFeature([], tf.string), 
    'WAVEFORM/feature/floats': tf.io.FixedLenFeature([], tf.string)
}

@tf.function
def _parse_function(example_proto):

    #Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    
    video_path = example['video_path']
    video = tf.io.decode_raw(example['image/encoded'], tf.int8)    
    spectrogram = tf.io.decode_raw(example['WAVEFORM/feature/floats'], tf.float32)
    
    label = example["clip/label/text"]
    label_map = example["clip/label/index"]
    
    return video, spectrogram, label_map

@tf.function
def decode_inputs(video, spectrogram, label_map):
    '''Decode tensors to arrays with desired shape'''

    frame = tf.reshape(video, [10, 3, 256, 256])
    frame = frame[0] #Pick the first frame.

    label_map = tf.expand_dims(label_map, axis = 0)
    
    sample = {'video_reshaped': frame, 'spectrogram': spectrogram, 'label_map': label_map}
    return sample

class FakeAVCelebDataset:

    def __init__(self, args):
        self.args = args
        self.samples = self.load_features_from_tfrec()

    def load_features_from_tfrec(self):
        '''Loads raw features from a tfrecord file and returns them as raw inputs'''
        ds = tf.io.matching_files(self.args.data_dir)
        files = tf.random.shuffle(ds)

        shards = tf.data.Dataset.from_tensor_slices(files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.map(_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.map(decode_inputs, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size = self.args.batch_size)

        return dataset
    
    def __len__(self):
        self.samples = self.load_features_from_tfrec(self.args.data_dir)
        cnt = self.samples.reduce(np.int64(0), lambda x, _: x + 1)
        cnt = cnt.numpy()
        return cnt
