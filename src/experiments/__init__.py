"""
This Module provides scripts and methods for inference.

The **word_rec** module deals with the evaluation of the model for the task of word recognition.
The **visualize_nn_progress** module provides a script to evaluate all model-states in a directory, writing the results into a single JSON file.
This can be of use for visualization of the neural nets progress.
Simply pass the output directory of the logged states during training to the *net_path* option.

|

Example to run word recognition:

::

    python3 src/experiments/word_rec.py \\
    path/to/state_dict \\
    path/to/dir_out \\
    dset_name \\
    path/to/dset_annotations \\
    path/to/imgs \\
    --gpu_idx=cuda:0 \\
    --estimator=cosine

|

"""


import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
