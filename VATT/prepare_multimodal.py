import yaml
import numpy as np
import tensorflow as tf
from models import multimodal 
from models.audio import audio_resnet as audio_backbone
from models.video import i3d as video_backbone
from models.text import bert_text as bert_backbone
from models import model_factory as multimodal_factory

with open('config.yaml', 'r') as f:
    configurations = yaml.load(f, Loader=yaml.FullLoader)

i3d = video_backbone.InceptionI3D()

audio_encoder = audio_backbone.Resnet()
# video_encoder = video_backbone.VideoModel(base_model = i3d)
text_encoder = bert_backbone.BertEncoder()

"""
 Args:
      video: The videos tensor of shape [B1, T, H, W, 3] where B1 is the batch
        size, T is the number of frames per clip, H the height, W the width
        and 3 the rgb channels.
      audio: The audio tensor of shape [B2, T', F] where B2 is the
        batch size, T' is the number of temporal frames, F is the number of
        frequency frames.
      word_ids: If words_embeddings is set to None, it will use the word indices
        input instead so that we can compute the word embeddings within the
        model graph. The expected shape is [B3, N, D] where B3 is the batch size
        and N the maximum number of words per sentence.
"""
# ATVF = multimodal_factory.vatt_models.AudioTextVideoFusion(audio_backbone=audio_encoder,
#                                                            video_backbone=video_encoder,
#                                                            text_backbone=text_encoder)
video_data = np.random.rand(10, 300, 300, 3)
audio_data = np.random.rand(4, 2)
text_data = np.random.rand(32, 20, 1)

inputs = {'video': video_data,
          'audio': audio_data,
          'text': text_data}

# audio_embeddings = audio_encoder(inputs['audio'])
# print(audio_embeddings.shape)

print(audio_encoder.input)
# outputs = ATVF(video = inputs['video'],  audio = inputs['audio'], word_ids = inputs['text'], training = False)
# print(outputs.shape)
# model = multimodal_factory.build_model(params = configurations)

# print(model)

# for layer in model.layers:
#     print(layer)



# output = model(inputs, training = False)
# print(output)
# # input_data = tf.keras.Input(shape = )
# # print(model.summary())

