import torch
from PIL import Image
import numpy as np
from vgg_model import VGGNormal, VGGDepthNormal
from dataset import load_data
import sys


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


def result_print(predictor, dataloader, file, has_depth=False):
    mean_err = 0
    median_err_arr = []
    less_11_25 = 0
    less_22_5 = 0
    less_30 = 0
    total_pixels = 0
    i = 1
    with torch.no_grad():
        for data in dataloader:
            if i % 100 == 0:
                print('Evaluating %d/%d' % (i, len(dataloader)))
            inputs = data['image']
            normals = data['normal']
            depths = data['depth']
            inputs, normals, depths = inputs.cuda(), normals.cuda(), depths.cuda()
            if has_depth:
                outputs = predictor(inputs, depths)
            else:
                outputs = predictor(inputs)

            err = angular_error(normals.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                                outputs.cpu().squeeze(0).permute(1, 2, 0).numpy())
            tmp_err = np.mean(err)
            if mean_err > 0:
                mean_err = (mean_err + tmp_err) / 2
            else:
                mean_err = tmp_err
            total_pixels += err.shape[0]

            median_err_arr.append(np.median(err))
            less_11_25 += np.count_nonzero(err < 11.25)
            less_22_5 += np.count_nonzero(err < 22.5)
            less_30 += np.count_nonzero(err < 30)
            i += 1

        median = np.median(np.array(median_err_arr))

        file.write("Mean = " + str(mean_err) + "\n")
        file.write("Median = " + str(median) + "\n")

        file.write("<11.25 = " + str(less_11_25 / total_pixels) + "\n")
        file.write("<22.5 = " + str(less_22_5 / total_pixels) + "\n")
        file.write("<30 = " + str(less_30 / total_pixels) + "\n")


if __name__ == "__main__":
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 2:
        print("Missing model location as first argument. Optional argument [--depth,-d] to evaluate VGG16 based model with depth map interpretation.")
        exit(1)
    model = sys.argv[1]
    if "-d" in sys.argv or "--depth" in sys.argv:
        depth = True
    else:
        depth = False

    f = open("results.txt", "a")
    f.write(f"{model}\n")
    if depth:
        predictor = VGGNormal().cuda()
    else:
        predictor = VGGDepthNormal().cuda()

    predictor.load_state_dict(torch.load(model, weights_only=True))
    predictor.eval()

    train_dataloader, test_dataloader = load_data('Data', 1, depth=True)
    train_indoor_dataloader, test_indoor_dataloader = load_data('Data', 1, depth=True, indoor=True,
                                                  outdoor=False)
    train_outdoor_dataloader, test_outdoor_dataloader = load_data('Data', 1, depth=True, indoor=False,
                                                                outdoor=True)

    f.write("DIODE; ALL DATA; TRAIN \n")
    result_print(predictor, train_dataloader, f, depth)
    f.write("DIODE; ALL DATA; TEST \n")
    result_print(predictor, test_dataloader, f, depth)

    f.write("DIODE; INDOOR DATA; TRAIN \n")
    result_print(predictor, train_indoor_dataloader, f, depth)
    f.write("DIODE; INDOOR DATA; TEST \n")
    result_print(predictor, test_indoor_dataloader, f, depth)

    f.write("DIODE; OUTDOOR DATA; TRAIN \n")
    result_print(predictor, train_outdoor_dataloader, f, depth)
    f.write("DIODE; OUTDOOR DATA; TEST \n")
    result_print(predictor, test_outdoor_dataloader, f, depth)
    f.write("\n-------------------------------------------------\n\n")
    f.close()
