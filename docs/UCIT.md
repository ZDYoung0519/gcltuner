## UCIT Benchmark
### 1. Download UCIT Instructions
Download UCIT Instructions: 
```bash
mkdir $YOUR_DATASET_ROOT/UCIT
cd $YOUR_DATASET_ROOT/UCIT
huggingface-cli download HaiyangGuo/UCIT --repo-type dataset --local-dir ./
```

And Organize the instructions as follows: 
```
UCIT/
|-- ArxivQA
|   |-- test_3000.json
|   `-- train_4w.json
|-- CLEVR-Math
|   |-- test_3000.json
|   `-- train_4w.json
|-- Flickr30k
|   |-- test_3000.json
|   |-- train_brief_4w.json
|   `-- val_coco_type_3000.json
|-- IconQA
|   |-- test_3000.json
|   `-- train.json
|-- ImageNet-R
|   |-- test_3000.json
|   `-- train.json
|-- VizWiz
|   |-- test_3000.json
|   |-- train.json
|   `-- val_coco_type_3000.json
```

### 2. Download UCIT Images
Then you need to download the images for each task:
|Image Source | Download Path|
| :-: | :-: |
|ArxivQA|[images](https://huggingface.co/datasets/MMInstruction/ArxivQA/tree/main)|
|ImageNet-R|Provided in UCIT/Imaget-R|
|IconQA|[images](https://iconqa.github.io/)|
|CLEVR-Math|[images](https://huggingface.co/datasets/dali-does/clevr-math/tree/main)|
|VizWiz|[images](https://vizwiz.org/tasks-and-datasets/image-captioning/)|
|Flickr30k|(Provided in UCIT/Flickr30k)

And organize them as follows:

```
|-- DATASET_ROOT
    |-- ArxivQA
        |-- images/
    |-- CLEVR
        |-- images
            |-- train/
            |-- test/
            |-- val/
    |-- Flickr30k
        |-- train/
        |-- val/
    |-- IconQA
        |-- iconqa_data/
            |-- iconqa/
    |-- ImageNet-R
        |-- train/
        |-- test/
    |-- VizWiz
        |-- train/
        |-- test/
        |-- val/
```

### 3. Modify the config file
Remberember to to modify the `data_root` to your instrutions path, `image_folder` to your image folder in each config file (e.g., `experimetns/exp_name/config.py`).



### 4. Preporess for text data (Optional)
We also recommand you to preprocess the text data with llama tokenizer (which is adopted by LLaVA):
```
python gcltuner/tools/precess_text_tokens_for_llava.py projects/lora/configs/ucit_vicuna_7b_v15_clip_vit_large_p14_336_none.py --save-folder $PROCESSED_PATH
```

Finally, you need to modify `data_root_ucit_offline` to `$PROCESSED_PATH` in the `gcltuner/data.py` file.

If you don't to use them, you need to set `data_root_ucit_offline` to `None`.



