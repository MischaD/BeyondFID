import ml_collections
from torchvision import transforms as T


config = ml_collections.ConfigDict()



# feature extraction configs 
config.feature_extractors = feature_extractors = ml_collections.ConfigDict()
feature_extractors.names = "inception,byol,random"

feature_extractors.byol = byol = ml_collections.ConfigDict()
byol.name = "byol" # necessary for all feature extractors
byol.batch_size = 256 # necessary for all feature extractors
byol.input_size = 96 # necessary for all feature extractors
byol.config = ml_collections.ConfigDict()
byol.config.path = "external/PyTorch-BYOL/resnet-18_80-epochs/checkpoints/"

feature_extractors.inception = inception = ml_collections.ConfigDict()
inception.name = "inception" # necessary for all feature extractors
inception.batch_size = 256
inception.input_size = 299
inception.config = ml_collections.ConfigDict()
inception.config.resize_input = True 
inception.config.normalize_input = True

feature_extractors.random = random = ml_collections.ConfigDict()
random.name = "random"
random.batch_size = 256
random.input_size = 299
random.config = ml_collections.ConfigDict()
random.config.resize_input = True 
random.config.normalize_input = True