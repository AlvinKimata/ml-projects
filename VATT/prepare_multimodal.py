import yaml
import numpy as np
import tensorflow as tf
# from models import multimodal 
from models import model_factory as multimodal_factory
from models import multimodal as vatt_models
from models.heads import factory as head_factory


with open('config.yaml', 'r') as f:
    configurations = yaml.load(f, Loader=yaml.FullLoader)


video_data = np.random.rand(10, 300, 300, 3)
audio_data = np.random.rand(10, 48_000)
text_data = np.random.rand(10, 20, 1)

inputs = {'video': video_data,
          'audio': audio_data,
          'word_ids': text_data}

video_shape = video_data.shape[1:]  # Exclude the batch size
audio_shape = audio_data.shape[1:]  # Exclude the batch size
text_shape = text_data.shape[1:]    # Exclude the batch size

video_input = tf.keras.Input(shape=video_shape)
audio_input = tf.keras.Input(shape=audio_shape)
text_input = tf.keras.Input(shape=text_shape)


# Instantiate the UnifiedFusion model
univatt = vatt_models.AudioTextVideoFusion()

# Pass the input tensors through the UnifiedFusion model
outputs = univatt(video=video_input, audio=audio_input, word_ids=text_input)

# Create the functional model
model = tf.keras.Model(inputs=[video_input, audio_input, text_input], outputs=outputs)

print(model)