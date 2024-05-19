import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np

from utils.datasets import *


if __name__=="__main__":
    imroot=''


    dataset = LoadImagesAndLabels(path, 640, 1,
                                    augment=False,  # augment images
                                    hyp=None,  # augmentation hyperparameters
                                    rect=False,  # rectangular training
                                    cache_images=False,
                                    single_cls=True
                                    stride=int(stride),
                                    pad=pad,
                                    image_weights=image_weights,
                                    prefix=prefix,
                                    tidl_load=tidl_load,
                                    kpt_label=kpt_label)


    print('finish')