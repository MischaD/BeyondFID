import ml_collections
from torchvision import transforms as T


config = ml_collections.ConfigDict()
config.master_port = 12344
config.basedir = "" # automatically extracts it from the Filepath

feature_models = "inception,byol"#,byol,random,dinov2,clip" # all of these features will be computed
config.metric_list = "irs,fid,prdc,is_score,cttest,authpct,fld,kid,vendi"
config.num_workers = 0 # for video data use low values, for images high (vid:0-2, img:4-8)

# feature extraction configs 
config.feature_extractors = feature_extractors = ml_collections.ConfigDict()
feature_extractors.names = feature_models 
config.feature_extractors.always_overwrite_snth = False # ignores if hash of synthetic dataset exists -> always recomputes features

feature_extractors.byol = byol = ml_collections.ConfigDict()
byol.name = "byol" # necessary for all feature extractors
byol.batch_size = 32 # necessary for all feature extractors
byol.config = ml_collections.ConfigDict()
byol.config.model_path = "./byol/large_model.pth" # relative to os.path.dirname(..-byol.py)
byol.config.cfg_path = "./byol/config_large.yaml"

feature_extractors.inception = inception = ml_collections.ConfigDict()
inception.name = "inception" # necessary for all feature extractors
inception.batch_size = 64 
inception.config = ml_collections.ConfigDict()
inception.config.resize_input = True 
inception.config.normalize_input = True

feature_extractors.random = random = ml_collections.ConfigDict()
random.name = "random"
random.batch_size = 64 
random.config = ml_collections.ConfigDict()
random.config.resize_input = True 
random.config.normalize_input = True

feature_extractors.dinov2 = dinov2 = ml_collections.ConfigDict()
dinov2.name = "dinov2"
dinov2.batch_size =  64 
dinov2.config = ml_collections.ConfigDict() # pass to constructor here

feature_extractors.convnext = convnext = ml_collections.ConfigDict()
convnext.name = "convnext"
convnext.batch_size = 64 
convnext.config = ml_collections.ConfigDict() # pass to constructor here
convnext.config.arch = "convnext_xlarge.fb_in22k"

feature_extractors.mae = mae = ml_collections.ConfigDict()
mae.name = "mae"
mae.batch_size = 16 
mae.config = ml_collections.ConfigDict() # pass to constructor here
mae.config.checkpoint = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"

feature_extractors.data2vec = data2vec = ml_collections.ConfigDict()
data2vec.name = "data2vec"
data2vec.batch_size = 16 
data2vec.config = ml_collections.ConfigDict() # pass to constructor here
data2vec.config.checkpoint = "facebook/data2vec-vision-large"

feature_extractors.swav = swav = ml_collections.ConfigDict()
swav.name = "swav"
swav.batch_size = 64 
swav.config = ml_collections.ConfigDict() # pass to constructor here

feature_extractors.clip = clip = ml_collections.ConfigDict()
clip.name = "clip"
clip.batch_size = 64 
clip.config = ml_collections.ConfigDict() # pass to constructor here

feature_extractors.flatten = flatten = ml_collections.ConfigDict()
flatten.name = "flatten"
flatten.batch_size = 64 
flatten.config = ml_collections.ConfigDict() # pass to constructor here

feature_extractors.flatten_resize = flatten_resize = ml_collections.ConfigDict()
flatten_resize.name = "flatten_resize"
flatten_resize.batch_size = 64 
flatten_resize.config = ml_collections.ConfigDict() # pass to constructor here
flatten_resize.config.resize_to = 224

feature_extractors.sdvae = sdvae = ml_collections.ConfigDict()
sdvae.name = "sdvae"
sdvae.batch_size = 32
sdvae.config = ml_collections.ConfigDict() # pass to constructor here
sdvae.config.path = "stabilityai/stable-diffusion-2"


# config.metrics 
config.metrics = metrics = ml_collections.ConfigDict()
metrics.fid = fid = ml_collections.ConfigDict()
fid.models = feature_models

metrics.prdc = prdc = ml_collections.ConfigDict()
prdc.models = feature_models
prdc.nearest_k = 5

metrics.authpct = authpct = ml_collections.ConfigDict()
authpct.models = feature_models

metrics.cttest = cttest = ml_collections.ConfigDict()
cttest.models = feature_models

metrics.fld = fld = ml_collections.ConfigDict()
fld.models = feature_models

metrics.kid = kid = ml_collections.ConfigDict()
kid.models = feature_models

metrics.diversity = diversity = ml_collections.ConfigDict()
diversity.models = feature_models
diversity.alphas = [2, 4]
diversity.batch_size = 512*(2**2)
diversity.folds = 3 # how often to compute with different subset. Only used if train and test is the same

metrics.irs = irs = ml_collections.ConfigDict()
irs.models = feature_models
irs.alpha_e = 0.05
irs.prob_tolerance = 1e-6
irs.naive = False # binary search or naive
irs.batch_size = 512*(2**2)
irs.verbose = True

metrics.vendi = vendi = ml_collections.ConfigDict()
vendi.models = feature_models
vendi.q=1
vendi.normalize=True
vendi.kernel='linear'
vendi.max_size=10_000

metrics.is_score = is_score = ml_collections.ConfigDict()
is_score.models = "inception" # -- hardcoded right now 
is_score.splits = 10 