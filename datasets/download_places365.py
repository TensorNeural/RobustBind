from torchvision import datasets

# For the validation set
datasets.Places365(root='Places365/', split='val', download=True)

# For the standard training set
# datasets.Places365(root='Places365/', split='train-standard', download=True)
