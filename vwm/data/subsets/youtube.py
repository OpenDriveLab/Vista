import os

from .common import BaseDataset


class YouTubeDataset(BaseDataset):
    def __init__(self, data_root="data/YouTube", anno_file="data/YouTube_vista.json",
                 target_height=320, target_width=576, num_frames=25):
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        print("YouTube loaded:", len(self))

    def get_image_path(self, sample_dict, current_index):
        first_frame = sample_dict["first_frame"]
        idx_str, ext_str = first_frame.split(".")
        format_length = len(idx_str)
        start_index = int(idx_str)
        file_name = str(start_index + current_index).zfill(format_length) + "." + ext_str
        return os.path.join(self.data_root, sample_dict["folder_name"], file_name)
