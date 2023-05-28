import os
from imutils import paths as path
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

        self.paths = []
        self.video_paths = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            cls_video_dir = sorted(path.list_files(cls_dir))
            
            for filename in cls_video_dir:
              filename = str(filename)
              if filename.endswith('.avi'):
                video_directory = filename

                audio_id = filename.split(sep = '.')[0]
                audio_id_dir = os.path.split(video_directory)[0]
                audio_dir_id = audio_id_dir.split(sep = '/')[-1]

                audio_path =  audio_id + '.wav'
                audio_file = os.path.split(audio_path)[-1]
                audio_directory = os.path.join(root_dir, 'audio', audio_dir_id, audio_file)
                self.paths.append((video_directory, audio_directory, cls))
          
        #Split dataset.
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        video_path, audio_path, data_class = self.paths[index]

        audios = data.load_and_transform_audio_data([audio_path], self.device)
        videos = data.load_and_transform_video_data([video_path], self.device)
        text = data.load_and_transform_text([data_class], self.device)

        return videos, ModalityType.VISION, text, ModalityType.TEXT
