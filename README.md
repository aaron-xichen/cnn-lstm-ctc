# Theano implementation of LSTM and CTC
- Support GPU accelaration. Please pay attention that only when the module is complicated enough the GPU effectiveness can be seen
- Support LogCTC, which prevents from overflow issue
- Support batch mode, which means that different size of image are allowed in a single batch

# Data format
- **x** is a list, each of which is H x W image. Here H should be identical, while W varies.
- **x_mask** is a matrix, each row of which means the valid region of corresponding image, thus is a mask. For example, `[[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]` means the width of first image is 3 and the second is 2
- **y** is a matrix, each row of which is the labels
- **y_clip** is a vector, each element of which means the length of corresponding label

Please refer to `prepare_testing_data` and `prepare_traing_data` functions in src/utee.py for more details

# Installation

Ubuntu:

```
sudo apt-get update
sudo apt-get install -y libmagickwand-dev python-opencv
pip install -r requirements.txt
```

Mac OS X:

```
brew tap homebrew/science
brew install opencv
pip install -r requirements.txt
```

# Usage
Once the data is prepared, simplily run `python solver.py` to start the training procedure

# Daemon
**Daemon** contains online version of feedforward stage, which means you can simply invoke the API to recognize a given images, with given model path.

# Others
My own experiments shows that 92.8% accuracy can be obtained if trained on andequate training samples, say, 20W. In my own case, images are simply handwrite english words.
