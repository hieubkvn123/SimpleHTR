from path import Path
from dataloader_iam import DataLoaderIAM

# Initialization
data_dir = "../data/IAM"
batch_size = 16
fast = False

# Build the dataloader
loader = DataLoaderIAM(Path(data_dir), batch_size, fast=fast)

# Get sample batch
# Now the idea is to create a similar dataloader for fake data
img, text, _ = loader.get_next()
