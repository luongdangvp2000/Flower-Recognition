import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    predict_image,
    test_predict_image
)

if __name__ == "__main__":
    img, label = valid_ds[6]
    plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
