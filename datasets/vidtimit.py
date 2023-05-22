import os
from typing import Optional, Callable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from models.imagebind_model import ModalityType
import data

class VIDTIMITDataset(Dataset):
    def __init__(self, root_dir : str, transform : Optional[Callable] = None,
    split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths_tuple = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            cls_audio_dir = os.path.join(cls_dir, 'audio')
            cls_video_dir = os.path.join(cls_dir, 'videos')
            cls_audio_files = [f for f in os.listdir(cls_audio_dir) if f.endswith('.wav')]
            cls_video_files = [f.replace('.wav', '.avi') for f in cls_audio_files]

            self.paths_tuple.extend(zip(cls_audio_files, cls_video_files))

        #Split dataset.
        train_paths, test_paths = train_test_split(self.paths_tuple, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        audio_path, class_text = self.paths_tuple[lambda x: x[index][0]]
        audios = data.load_and_transform_audio_data([audio_path], self.device, to_tensor=False)

        video_path, class_text = self.paths_tuple[lambda x: x[index][1]]
        videos = data.load_and_transform_video_data([video_path], self.device, to_tensor = False)

        if self.transform is not None:
            video = videos[0]
            transformed_videos = self.transform(video)

        if self.transform is not None:
            audio = audios[0]
            transformed_audios = self.transform(audio)

        texts = data.load_and_transform_text([class_text], self.device)

        return transformed_videos, ModalityType.VISION, transformed_audios, ModalityType.VISION, texts, ModalityType.TEXT