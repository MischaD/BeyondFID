import ml_collections
from torchvision import transforms as T


config = ml_collections.ConfigDict()


# feature extraction configs 
config.feature_extractors = feature_extractors = ml_collections.ConfigDict()
feature_extractors.names = "clip,dinov2,inception,byol,random"

feature_extractors.byol = byol = ml_collections.ConfigDict()
byol.name = "byol" # necessary for all feature extractors
byol.batch_size = 256 # necessary for all feature extractors
byol.config = ml_collections.ConfigDict()
byol.config.path = "beyondfid/feature_extractor_models/byol"

feature_extractors.inception = inception = ml_collections.ConfigDict()
inception.name = "inception" # necessary for all feature extractors
inception.batch_size = 256
inception.config = ml_collections.ConfigDict()
inception.config.resize_input = True 
inception.config.normalize_input = True

feature_extractors.random = random = ml_collections.ConfigDict()
random.name = "random"
random.batch_size = 256
random.config = ml_collections.ConfigDict()
random.config.resize_input = True 
random.config.normalize_input = True

feature_extractors.dinov2 = dinov2 = ml_collections.ConfigDict()
dinov2.name = "dinov2"
dinov2.batch_size = 256
dinov2.config = ml_collections.ConfigDict() # pass to constructor here

feature_extractors.clip = clip = ml_collections.ConfigDict()
clip.name = "clip"
clip.batch_size = 256
clip.config = ml_collections.ConfigDict() # pass to constructor here


# config.metrics 
config.metrics = metrics = ml_collections.ConfigDict()
metrics.fid = fid = ml_collections.ConfigDict()
fid.model = "inception,byol,random,dinov2,clip"

metrics.prdc = prdc = ml_collections.ConfigDict()
prdc.model = "inception,byol,random,dinov2,clip"
prdc.nearest_k = 1