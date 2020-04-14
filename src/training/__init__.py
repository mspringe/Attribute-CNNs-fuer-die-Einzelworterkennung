"""
This module provides scripts to train the PHOCNet and the RCCA estimator.

|

Example for training the PHOCNet:

::

    python3 src/training/phocnet_trainer.py \\
    path/to/output_dir/ \\
    gw \\
    /path/to/gw_database/almazan/queries/queries.gtp \\
    /path/to/gw_database/almazan/images \\
    --max_iter=1e5 \\
    --model_name=my_PHOCNet \\
    --gpu_idx=cuda:0 \\
    --k_fold=1 \\
    --alphabet=ldp \\
    --s_batch=10

|

Example for training the RCCA:

::

    python3 src/training/cca_cross_validation.py \\
    /path/to/dir_out \\
    gw \\
    /path/to/gwdb/almazan/queries/queries.gtp \\
    /path/to/gwdb/almazan/images \\
    /path/to/nn_my_PHOCNet.pth \\
    --k_fold=1 \\
    --model_name=my_RCCA \\
    --gpu_idx=cuda:0 \\
    --alphabet=ldp

|

see also :func:`src.parser.args_parser.parser_training` for all options, regarding training.
"""
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
