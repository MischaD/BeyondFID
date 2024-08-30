# BeyondFID
A python package to streamline evaluation of unconditional image generation models. 


## installation 

    pip install beyondfid

## Prepearation 
There are two potential ways to define datasets:  

1. CSV File:
    FileName, Split
    TRAIN, VAL, TEST

2. Folders:
    a) Train/
          *.png
       Test/..
          *.png
    b) ./*.png 

    allowed file extensions are: png, jpg, pt, mp4, avi

## Rules for Video as input data

Framewise, accumulation strategy can be defined in config


## Usage 

As CLI 

    python beyondfid --config "default.py"\
                     --pathreal "path/to/real/" --pathfake "path/to/fake"\ 
                     --feature_extractor "inception"\
                     --output_path "path/to/output"\
                     --results_filename "results.json"\
                     --no_inception_score # skips computation of inception score

                    # adjust config: 
                    "--config-update=metrics.fid.model=random,feature_extractors.names=fid"

or within python 

    import beyondfid
    config = beyondfid.load_config("default.py")
    beyondfid.run(config, pathreal, pathfake, feature_extractor, output_path, results_filename)

## Requirements 

CUDA has to be available and it needs to run on a GPU 

## Acknowledgements 
This work is based on the work of the following repositories:


https://github.com/clovaai/generative-evaluation-prdc

https://github.com/mseitzer/pytorch-fid

## How to add a new model: 

We show the example of adding clip to the list of feature extractors. Create a file in beyondfid/feature_extractor_models/clip.py

Import base class and the register hook: 

    from . import BaseFeatureModel, register_feature_model

Define new class, import Base classes, and implement compute_latent(x) -- (model_config will be explained in the next step): 

    @register_feature_model(name="clip")
    class CLIP(BaseFeatureModel, nn.Module):
        def __init__(self, model_config):
            super().__init__()
            # transform? input is B x 3 x H x W from 0 to 1 (RBG)

        def compute_latent(self, x):
            y = x#self.model.encode_image(x)
            return y

Register the module by importing it in beyondfid/feature_extractor_models/__init__.py at the end of the fileo

    from beyondfid.feature_extractor_models.clip import CLIP 

Adjust the config (config.py) as necessary. To add the computation to the run.py method add the following line to config.py (has to be the same as name)

    feature_extractors.names = "clip"

The same for all metrics you want to compute. Just append it to the list: 

    fid.model = "inception,clip"

Next put the settings of the feature extractor in config.py - 

    feature_extractors.clip = clip = ml_collections.ConfigDict()
    clip.name = "clip"
    clip.batch_size = 256
    clip.config = ml_collections.ConfigDict() # pass to constructor here

clip.config will be parsed as model_config to the constructor of the CLIP class if you want to test different settings.
