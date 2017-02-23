# Implementation of MVFusionNet on Torch #

The framework is trained and tested on ShapeNetCore55 for 3D shape recognition and retrieval. It consists of two networks, a ResNet for deep feature extraction and a Fusion Network which fuses the extracted deep features with hand-crafted one.

# Dataset #
The framework is trained and evaluated on ShapeNetCore55 benchmark

# Dependencies #
* cunn `luarocks install cunn`
* cudnn `luarocks install cudnn`
* Download ResNet (resnet-18 used) to the diretory Network1/models/resnet

# Run #
Network1 and Network2 run seperately. Run Network1 to extract deep features per rendered view of a 3D shape. Run Network2 to fuse your hand-crafted features with the previously extracted deep ones. For retrieval remove the last Linear layer (classification layer) to get the 3D descriptor.

### Network1 ###
`cd/Network1`
`th main.lua [options]`
### Network2 ###
`cd/Network2`
`th main.lua [options]`

# Examples #

### Network1 ###
Training example: place your training data in a directory named trainSet (each class is a subdirectory) and train the model with batch size 128 for your e.g. 224x224 images.
`th main.lua -mode train -inputDataPath /path/to/sets -dirName trainSet -batchSize 128 -imageSize 224`

Feature extraction example: choose a name for the directory where the 't7' files will be extracted. Choose the set that you want to pass through the model (e.g. valSet). The 't7' will be a vector with Nx512 size where N is the number of the rendered views of the 3D shape.
`th main.lua -mode test -inputDataPath /path/to/sets -dirName valSet -targetDirName t7/val_features -batchSize 128 -imageSize 224`
