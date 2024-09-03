# BeyondFID
A python package to streamline evaluation of unconditional image generation models. 
If you have a folder full of images or videos and want to compute image-wise generative metrics you are at the right place. 
Supported metrics are:  TODO add references

    - FID
    - KID
    - FLD
    - AuthPCT
    - CTTest
    - Precision
    - Recall
    - Coverage 
    - Density
    - ReIDensity 
    - Inception Score

## installation 

    git clone git@github.com:MischaD/BeyondFID.git
    pip install -e .

## Preparation 
There are two potential ways to define datasets:  

1. **CSV File Structure:**

   | FileName            | Split        |
   |---------------------|--------------|
   | path/to/file.png    | {TRAIN, VAL, TEST} |
   | l1/example_000.png  | TRAIN        |
   | l1/any/path/example_001.png  | TEST         |
   | l2/example_001.png  | TRAIN        |

    The filename describes the relative path from the csv to the file. If you use ./data/FileList.csv as argument for pathtrain then it expects the image to be in ./data/l1/example_000.png

2. **Folders:**

    If your dataset looks like this: 

        data/
        ├── train/
        │   ├── example000.png
        │   └── folder1/
        │       └── example001.png
        └── test/
            ├── example100.png
            └── folder1/
                └── example101.png

    then you can use the path to the folder directly **./data/train** and **./data/test**. It will automatically look for all files recursively.

**Allowed File Extensions:**
   - **Images:** `png`, `jpg`, `pt`
   - **Videos:** `mp4`, `avi`, `pt`

If the file extension is `pt` it will check for the number of dimensions and assumes that a three dimensional tensor is an image and a four dimensional one a video. 

## Rules for Video as input data
For videos we reduce the video to the first frame. This approach should only be used if the subject of the video does not change (For example a talking face).

## Usage 

As CLI 

    beyondfid path/to/train path/to/test path/to/synth
              --config "default.py"\
              --feature_extractor "inception"\
              --output_path "path/to/output"\
              --results_filename "results.json"\
              # overwrite any value from config.py: 
              "--config-update=metrics.fid.model=random,feature_extractors.names=fid"

or within python 

    from beyondfid.default_config import config 
    from beyondfid.run import run

    metrics="fid,is_score,kid" # decide what metrics you want to compute 

    config.feature_extractors.names="inception,dinov2" # decide what features to extract
    config.metrics.fid.model="inception,dinov2" # decide on what features you want to compute FID  
    results = run("path/to/train", "path/to/train", "path/to/synth", metrics, "out/path", "results.json", config)
    
    print(results["dinov2"]["fid_train"]) # fid with dino features between synthetic and train dataset
    print(results["inception"]["fid_train"])



## Requirements 

CUDA has to be available and it needs to run on a GPU 

## Acknowledgements 
This work is based on the work of the following repositories:


https://github.com/clovaai/generative-evaluation-prdc

https://github.com/mseitzer/pytorch-fid

## How to add a new model: 

In a world of infinite feature extractors, where different runs can already have noticable effects on the performance ([example](https://arxiv.org/abs/1801.01973)), you might want to add your own feature extractor. There is a quick way and a general way. 
**First we show the quick and dirty way**: 


**The clean way** is to add the extractor like we did. We show this by the example of how we added clip to the list of feature extractors. 

1. Create a file clip.py in:

        cd BeyondFID
        beyondfid/feature_extractor_models/clip.py


2. Import base class and the register hook: 

        from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model

3. Define the new feature extractor class, import base classes, give it a name in the hook,  and implement compute_latent(x). Arguments can be parsed as model_config which will be explained in the next step: 

        @register_feature_model(name="clip")
        class CLIP(BaseFeatureModel, nn.Module):
            def __init__(self, model_config):
                super().__init__()
                # transform? input is B x 3 x H x W from 0 to 1 (RBG)

            def compute_latent(self, x):
                y = x#self.model.encode_image(x)
                return y

4. Register the module by importing it in beyondfid/feature_extractor_models/__init__.py at the end of the file:

        from beyondfid.feature_extractor_models.clip import CLIP 

5. Adjust the config (default_config.py) as necessary. To add the computation to the run.py method append the model to the list of feature extractors in default_config.py (has to be the same as name)

        feature_extractors.names = "clip" + ",inception..." # 

6. The same for all metrics you want to compute on the extracted features. Just append it to the list: 

        fid.model = "inception,clip"

7. Next put the settings of the feature extractor in default_config.py - 

        feature_extractors.clip = clip = ml_collections.ConfigDict()
        clip.name = "clip"
        clip.batch_size = 256
        clip.config = ml_collections.ConfigDict() # passed to constructor here

clip.config will be parsed as model_config to the constructor of the CLIP class if you want to test different settings.
