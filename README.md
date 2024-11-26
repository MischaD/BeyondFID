# BeyondFID
A python package to streamline evaluation of (unconditional) image generation models. 
If you have a folder full of images or videos and want to compute image-wise generative metrics you are at the right place. 
Supported metrics are: 

- [IRS](https://arxiv.org/abs/2411.16171)
- [FID](https://arxiv.org/abs/1706.08500)
- [KID](https://arxiv.org/abs/1801.01401)
- [FLD](https://arxiv.org/abs/2302.04440)
- [AuthPCT](https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf)
- [CTTest](http://arxiv.org/abs/2004.05675)
- [Precision, Recall](https://arxiv.org/abs/1806.00035)
- [Coverage, Density](https://arxiv.org/abs/2002.09797) 
- [Inception Score](https://proceedings.neurips.cc/paper_files/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)

# Table of Contents
- [Installation](#installation)
- [Supported Dataset Structure](#supported-dataset-structure)
  - [Folders](#folders)
  - [CSV File Structure](#csv-file-structure)
  - [Large Tensors](#large-tensors)
- [Usage](#usage)
  - [As CLI](#as-cli)
  - [Within Python](#within-python)
  - [Extract IRS Score](#extracting-irs-score)
- [Advanced Usage](#advanced-usage)
  - [How to Add a New Feature Extraction Model](#how-to-add-a-new-feature-extraction-model)
    - [The Quick Way](#the-quick-way)
    - [The Clean Way](#the-clean-way)
  - [How to Use a Generator Function Instead of Saving Models on Disk](#how-to-use-a-generator-function-instead-of-saving-models-on-disk)
- [Acknowledgements](#acknowledgements)
- [Cite Us](#cite-us)

## Installation 

    git clone git@github.com:MischaD/BeyondFID.git
    pip install -e .
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

## Supported Dataset Structure 
There are three potential ways to define datasets:  

1. **Folders:**

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

2. **CSV File Structure:**

    In case you have a weird file structure or your splits are defined in a .txt file you can use a .csv file to define the relative path from the csv to the file. Use the keys **FileName** and **Split** and the correct string (uppercase TRAIN, VAL, TEST) to specify the split. 

   | FileName            | Split        |
   |---------------------|--------------|
   | path/to/file.png    | {TRAIN, VAL, TEST} |
   | l1/example_000.png  | TRAIN        |
   | l1/any/path/example_001.png  | TEST         |
   | l2/example_001.png  | TRAIN        |

    For example if you have **./data/FileList.csv** as argument for pathtrain then it expects the image to be in ./data/l1/example_000.png

3. **Large Tensors**

    In case you do not want so save the images on disk have a look at *How to use a generator function instead of saving the models on disk*



**Allowed File Extensions:**
   - **Images:** `png`, `jpg`, `pt`
   - **Videos:** `mp4`, `avi`, `pt`

If the file extension is `pt` it will check for the number of dimensions and assumes that a three dimensional tensor is an image and a four dimensional one a video. 

### Rules for Video as input data
For videos we reduce the video to the first frame. This approach should only be used if the subject of the video does not change (For example a talking face). For more information see [https://arxiv.org/abs/2411.04956?](https://arxiv.org/abs/2411.04956?).

## Usage 

As CLI 

    beyondfid path/to/train path/to/test path/to/synth --feature_extractors swav dinov2 inception --metrics irs fid

    #--config "new_default.py"\ # see beyondfid.default_config.py for an example 
    #--output_path "path/to/output"\ # where the features and results_filename will be saved
    #--results_filename "results.json"\ # where the metrics will be saved. 
    "--config-update=metrics.fid.model=random,feature_extractors.names=fid" #overwrite any value from default_config.py 

or within python 

```python
from beyondfid.default_config import config 
from beyondfid.utils import update_config
from beyondfid.run import run

# decide what features to extract, run beyondfid -h for a full list, can be custom (see below)
feature_extractors="swav,inception,dinov2" 

 # decide what metrics you want to compute, run beyonfid -h for a full list
metrics ="fid,irs"

# update the config
config = update_config(config, metrics=metrics, feature_extractors=feature_extractors) 

# manually set hyperparameters if necessary - full list is in beyondfid.default_config.py
config.metrics.prdc.nearest_k = 3

# path/to/synth can also be a generator function (see below)
results = run("path/to/train", "path/to/train", "path/to/synth", "out/path", "results.json", config) 
# results will also be saved to "out/path/results.json", all features as tensors to out/path/<model>/*.pt

print(results["dinov2"]["fid_train"]) # fid with dino features between synthetic and train dataset
```

⚠️ **Important:** For every feature extractor beyondfid only computes the feature tensor once. To save it for future use it uses the hash of the relative path to *all* files in the given directory (ignoring the name of the basedir). This means that two datasets with the same filestructure, names, and number of files will also have the same hash (e.g. a folder with 100 samples named sample_0.png-sample_100.png). Double check for every real dataset (you will also be able to see it from the output). For the synthetic datasets beyondfid always recomputes the feature tensor so you dont have to find different names for the same synthetic dataset (e.g. you compare multiple checkpoints with the same script). You can deactivate this behaviour by setting 

    config.feature_extractors.always_overwrite_snth = False

### Extracting IRS Score 

The results will be safed in a .json file. To extract the adjusted irs score we suggest SWaV as explained in the paper (jsonfile["swav"]["irs_adjusted"]).  
If your training dataset is much smaller than your test data then you can use the score directly (jsonfile["swav"]["snth"]["irs_inf"]). 

# Advanced Usage
- You do not want to save the generated images on disc? 
- You want to evalute your own feature extraction method? 

Then see below! 


# How to add a new feature extraction model: 

In a world of infinite feature extractors, where different runs can already have noticable effects on the performance ([example](https://arxiv.org/abs/1801.01973)), you might want to add your own feature extractor. There is a quick way and a general way. 

## The quick way
**First we show the quick and dirty way** (Only support single GPU for now as multiprocessing needs to be pickleable): 

```python 
    from beyondfid.default_config import config 
    from beyondfid.run import run_generic
    import torch # only for torch.randn

    # a callable object, takes 0-1 normalized (BxCxHxW) -> (BxF) and computes features 
    forward_function = lambda x:  torch.randn((x.size()[0], 2048))

    # define metrics you want to compute 
    config.metric_list = "fid,kid,prdc" 

    # (optional) set metric hyperparams -- see beyondfid/default_config.py for a full list
    config.metrics.prdc.nearest_k = 3 


    # use output path to name model. <output_path>/generic/hashdata_generic_<hash>.pt
    results = run_generic("path/to/train", "path/to/test", "path/to/snth", forward_function, config, output_path="./.cache/")
    # Some metrics like FID, IS, KID, FLD do not need train and test data. Just set the same path twice. 
    print(results)
```

## The clean way
**The clean way** is to add the extractor like we did. We show this by the example of how we added clip to the list of feature extractors. 

1. Create a file clip.py in:

        cd BeyondFID
        touch beyondfid/feature_extractor_models/clip.py
        vim beyondfid/feature_extractor_models/clip.py


2. Import base class and the register hook: 

    ```python 
        from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model
    ```

3. Define the new feature extractor class, import base classes, give it a name in the hook,  and implement compute_latent(x). Arguments can be parsed as model_config which will be explained in the next step: 

    ```python 
        @register_feature_model(name="clip")
        class CLIP(BaseFeatureModel, nn.Module):
            def __init__(self, model_config):
                super().__init__()
                # transform? input is B x 3 x H x W from 0 to 1 (RBG)

            def compute_latent(self, x):
                y = x#self.model.encode_image(x)
                return y
    ```

4. Register the module by importing it in beyondfid/feature_extractor_models/__init__.py at the end of the file:

    ```python 
        from beyondfid.feature_extractor_models.clip import CLIP 
    ```

5. Next put the settings of the feature extractor in default_config.py - 

    ```python 
        feature_extractors.clip = clip = ml_collections.ConfigDict()
        clip.name = "clip"
        clip.batch_size = 256
        clip.config = ml_collections.ConfigDict() # passed to constructor here
    ```

    clip.config will be parsed as model_config to the constructor of the CLIP class if you want to test different settings.

6. Run 

    run.py data/train data/test data/snth --feature_extractors clip --metrics fid fld prdc 


# How to use a generator function instead of saving the models on disk

For online processing of metrics, hyperparameter sweeps etc. it might be favorable to compute the metrics without saving images to disk. 
We also provide a function for this. 

```python 
    from beyondfid.default_config import config 
    from beyondfid.utils import update_config
    from beyondfid.run import run 
    import torch # only for torch.zeros

    test_images = {"file_key_test":torch.zeros((100, 3, 512, 512))} # key will be used to save features as tensor -- needs unique name -- tensor ranging from 0 to 1 
    generated_images = {"file_key_gen":torch.zeros((100, 3, 512, 512))} 

    # update config with metrics you want to compute
    config = update_config(config, metrics="fid", feature_extractors="inception,dinov2") 

    results = run("path/to/train", test_images, generated_images, results_filename="results.json", output_path="./.cache/", config=config)
    #print(results["inception"]["fid_train"]) - results[<feature_extractor_name>][metric_name]
    print(results)
```

This package was optimized to compute multiple metrics at once, which means we need to save all generated images in memory. 
This can get memory heavy. 50000 three channel images of size 512x512 take up 147 GB of memory. 
Make sure you have that available. If not, consider saving images on disk instead. Features get reused by different metrics therefore we save the on disk. After execution you can remove files from *./.cache* if necessary.



## Acknowledgements 
This work is based on the work of the following repositories and would not have been possible without them:

- https://github.com/clovaai/generative-evaluation-prdc
- https://github.com/marcojira/fld
- https://github.com/mseitzer/pytorch-fid
- https://github.com/layer6ai-labs/dgm-eval

We would also like to acknowledge [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) who provide a similar package with a different set of models and metrics.
    - IRS metric!
    - compute and extract multiple features at once
    - safe features for further processing
    - easily add own models
    - import and run within your own program

## Citation 

If you find this code useful, please cite us: 

    @misc{dombrowski2024imagegenerationdiversityissues,
        title={Image Generation Diversity Issues and How to Tame Them}, 
        author={Mischa Dombrowski and Weitong Zhang and Sarah Cechnicka and Hadrien Reynaud and Bernhard Kainz},
        year={2024},
        eprint={2411.16171},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2411.16171}, 
    }

