import numpy as np
from xtuner.dataset.llava import LLaVADataset as XTunerLLaVADataset
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN

class LLaVADataset(XTunerLLaVADataset):
    def __init__(
        self,
        image_folder,
        image_processor,
        data_path=None,
        tokenizer=None,
        offline_processed_text_folder=None,
        max_dataset_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        pad_image_to_square=False,
        sample_ratio=1,
    ):
        super().__init__(
            image_folder, 
            image_processor, 
            data_path, 
            tokenizer,
            offline_processed_text_folder,
            max_dataset_length,
            dataset_map_fn,
            template_map_fn,
            max_length,
            pad_image_to_square
        )

        self.sample_ratio = sample_ratio
        if self.sample_ratio < 1.0:
            total_samples = len(self.text_data)
            sample_size = int(total_samples * self.sample_ratio)
            print("Total size: {:d}, sample size: {:d} ({:.2f}%)".format(total_samples, sample_size, sample_ratio *100))
            indexes = np.random.choice(total_samples, sample_size, replace=False).tolist()
            indexes.sort()
            self.text_data = [self.text_data[idx] for idx in indexes]

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        text = self.text_data[index]['conversations'][0]['value'].replace(DEFAULT_IMAGE_TOKEN, "").replace('\n', "")
        data_dict["text"] = text
        return data_dict
