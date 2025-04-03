import torch
from PIL import Image
import os
from glob import glob
import numpy as np


def angular_error(exp, pred):
    if type(exp) == Image.Image: exp = np.array(exp)
    if type(pred) == Image.Image: pred = np.array(pred)

    x1 = exp[..., 0]
    y1 = exp[..., 1]
    z1 = exp[..., 2]
    x2 = pred[..., 0]
    y2 = pred[..., 1]
    z2 = pred[..., 2]

    len2A = x1 * x1 + y1 * y1 + z1 * z1
    len2B = x2 * x2 + y2 * y2 + z2 * z2

    maskExp = len2A > 0
    maskPred = len2B > 0
    mask = maskExp & maskPred

    exp = exp.astype(np.float32)
    pred = pred.astype(np.float32)

    exp = exp[mask]
    pred = pred[mask]

    exp = exp / np.linalg.norm(exp, axis=-1, keepdims=True)
    pred = pred / np.linalg.norm(pred, axis=-1, keepdims=True)

    # !phi = arccos(\frac{a \cdot b}{||a|| ||b||}), where |a| = |b| = 1
    product = np.sum(exp * pred, axis=-1)

    phi = np.arccos(np.clip(product, -1.0, 1.0))
    return phi * 180 / np.pi


if __name__ == "__main__":
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO selection of Data folder from arguments
    Y = glob(os.path.join("Data/", 'val', "*", "*", "*", '*normal.npy'))
    X = glob(os.path.join("Data/", 'val', "*", "*", "*", '*.png'))

    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)

    mean_err = 0
    median_err_arr = []
    less_11_25 = 0
    less_22_5 = 0
    less_30 = 0
    total_pixels = 0
    i = 1
    for x,y in zip(X, Y):
        print("Processing: ", i, "/", len(X))
        in_img = Image.open(x)
        gt_img = np.load(y)
        predicted = predictor(in_img)

        err = angular_error(gt_img, predicted)
        total_pixels += err.shape[0]
        tmp_err = np.mean(err)
        mean_err = (mean_err + tmp_err) / 2

        median_err_arr.append(np.median(err))
        less_11_25 += np.count_nonzero(err < 11.25)
        less_22_5 += np.count_nonzero(err < 22.5)
        less_30 += np.count_nonzero(err < 30)
        i += 1


    median = np.median(np.array(median_err_arr))

    print("Mean = ", mean_err)
    print("Median = ", median)

    print("<11.25 = ", less_11_25 / total_pixels)
    print("<22.5 = ", less_22_5 / total_pixels)
    print("<30 = ", less_30 / total_pixels)

