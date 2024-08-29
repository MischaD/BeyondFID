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

