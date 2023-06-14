import yaml
import numpy as np
import tensorflow as tf
from models import model_factory as multimodal_factory
from models import multimodal as vatt_models
from models.heads import factory as head_factory
from models.unified import factory as unified_factory
from models.unified import uvatt


with open('config.yaml', 'r') as f:
    configurations = yaml.load(f, Loader=yaml.FullLoader)


video_data = np.random.rand(10, 300, 300, 3)
audio_data = np.random.rand(10, 48_000)
text_data = np.random.rand(10, 20, 1)

inputs = {'video': video_data,
          'audio': audio_data,
          'text': text_data}

# video_shape = video_data.shape[1:]  # Exclude the batch size
# audio_shape = audio_data.shape[1:]  # Exclude the batch size
# text_shape = text_data.shape[1:]    # Exclude the batch size

# video_input = tf.keras.Input(shape=video_shape)
# audio_input = tf.keras.Input(shape=audio_shape)
# text_input = tf.keras.Input(shape=text_shape)


# Create a model instance
base_model = uvatt.UniversalVATT()
batch_size = 8
video_seq_len = 10
audio_seq_len = 10
text_seq_len = 10
audio_dim = 2
video_dim = 224
txt_vocab_size = 20


# Create dummy variables for each modality (video, audio, text)
video_inputs = tf.random.normal(shape=(batch_size, video_seq_len, video_dim, video_dim, 3))
audio_inputs = tf.random.normal(shape=(batch_size, audio_seq_len, audio_dim))
text_inputs = tf.random.uniform(shape=(batch_size, text_seq_len), maxval=txt_vocab_size, dtype=tf.float32)

# Construct the inputs dictionary
inputs = {
    "video": video_inputs,
    "audio": audio_inputs,
    "text": text_inputs
}

# Call the model to get the outputs
outputs = base_model(inputs, training=False)

# Access the output tensors and their shapes
video_features_pooled = outputs["video"]["features_pooled"]
video_features = outputs["video"]["features"]
audio_features_pooled = outputs["audio"]["features_pooled"]
audio_features = outputs["audio"]["features"]
text_features_pooled = outputs["text"]["features_pooled"]
text_features = outputs["text"]["features"]

# Print the shapes of the output tensors
print("Video features pooled shape:", video_features_pooled.shape)
print("Video features shape:", video_features.shape)
print("Audio features pooled shape:", audio_features_pooled.shape)
print("Audio features shape:", audio_features.shape)
print("Text features pooled shape:", text_features_pooled.shape)
print("Text features shape:", text_features.shape)
