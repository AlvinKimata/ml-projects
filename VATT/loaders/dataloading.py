'''Data loading script from tfrecord to raw inputs'''
import tensorflow as tf

DS_DIR = 'fakeavceleb_tfrec\\'

feature_description = {
    'clip/label/index': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'WAVEFORM/feature/floats': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    
    #Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    label = example["clip/label/index"]
    spectrogram = tf.io.decode_raw(example['WAVEFORM/feature/floats'], tf.float32)    
    frame = tf.io.decode_raw(example['image/encoded'], tf.uint8)
    frame = tf.cast(frame, tf.float32)
    
    return frame, spectrogram, label

files = tf.io.matching_files('fakeavceleb-500*')
files = tf.random.shuffle(files)
shards = tf.data.Dataset.from_tensor_slices(files)
dataset = shards.interleave(tf.data.TFRecordDataset)
dataset = dataset.shuffle(buffer_size=1000)

dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

MAX_SEQUENCE_LENGTH = 55680
def decode_inputs(frame, spec, label):
    '''Decode tensors to arrays with desired shape'''
    frame = tf.reshape(frame, [224, 224, 3])
    
    spec = tf.reshape(spec, [435, 128])

    
    label = tf.expand_dims(label, axis = 0)
    label =  tf.one_hot(label, depth = 4)    
    return frame, spec, label