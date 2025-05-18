from dataset import load_data
import torch
import sys
import copy
from main import angular_error
import numpy as np
from vgg_model import VGGNormal, VGGDepthNormal
import torch.nn as nn


def evaluation(dataloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_err = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            if i % 100 == 0:
                print('Evaluating %d/%d' % (i, len(dataloader)))
            inputs = data['image']
            normals = data['normal']
            depths = data['depth']
            inputs, normals, depths = inputs.to(device), normals.to(device), depths.to(device)
            outputs = model(inputs)
            if normals.shape[0] == 1:
                err = angular_error(normals.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                                    outputs.cpu().squeeze(0).permute(1, 2, 0).numpy())
                tmp_err = np.mean(err)
                if total_err > 0:
                    total_err = (total_err + tmp_err) / 2
                else:
                    total_err = tmp_err
            else:
                gt_images = normals.cpu().squeeze(0).permute(0, 2, 3, 1).numpy()
                pred_images = outputs.cpu().squeeze(0).permute(0, 2, 3, 1).numpy()
                err_arr = np.zeros(gt_images.shape[0] + 1, dtype=float)
                for i in range(gt_images.shape[0]):
                    err_arr[i] = np.mean(angular_error(gt_images[i], pred_images[i]))

                err_arr[-1] = total_err
                if total_err > 0:
                    total_err = np.mean(err_arr)
                else:
                    total_err = np.mean(err_arr[:-1])

    return total_err


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 6:
        print("Not enough arguments. Required arguments are model [depth, vgg], batch size[int], learning rate[float], epochs[int], dataset[in,out,both]")
        exit(1)

    model_type = sys.argv[1]
    batch_size = int(sys.argv[2])
    lr = float(sys.argv[3])
    epochs = int(sys.argv[4])
    dataset = sys.argv[5]
    if model_type == 'depth':
        model = VGGDepthNormal().cuda()
    else:
        model = VGGNormal().cuda()
    if dataset == 'in':
        indoor_subset = True
        outdoor_subset = False
    elif dataset == 'out':
        indoor_subset = False
        outdoor_subset = True
    elif dataset == 'both':
        indoor_subset = True
        outdoor_subset = True
    else:
        print("Not a valid dataset")
        exit(1)


    train_dataloader, test_dataloader = load_data('Data', batch_size, depth=True, indoor=indoor_subset,
                                                  outdoor=outdoor_subset)
    loss_epoch_arr = []
    best_model = model.state_dict()

    max_epochs = 16
    min_loss = 1000000


    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_iters = len(train_dataloader)
    model.train()
    for epoch in range(max_epochs):
        for i, data in enumerate(train_dataloader, 0):
            inputs= data['image']
            normals = data['normal']
            depths = data['depth']
            inputs, normals, depths = inputs.to(device), normals.to(device), depths.to(device)

            opt.zero_grad()

            outputs = model(inputs,depths)
            loss = loss_fn(outputs, normals)
            loss.backward()
            opt.step()

            if min_loss > loss.item():
                min_loss = loss.item()
                best_model = copy.deepcopy(model.state_dict())
                print('Min loss %0.2f' % min_loss)

            if i % 100 == 0:
                print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

            del inputs, normals, outputs
            torch.cuda.empty_cache()

        loss_epoch_arr.append(loss.item())
        print('Epoch: %d/%d' % (epoch + 1, max_epochs))

    model.load_state_dict(best_model)
    model.eval()
    print("Evaluating Training subset dataset with the best model")
    eval_train = evaluation(train_dataloader, model)
    print("Evaluating Testing subset dataset with the best model")
    results_test = evaluation(test_dataloader, model)
    print('END SUBSET: Test acc: %0.2f, Train acc: %0.2f' % (results_test, eval_train))

    train_dataloader, test_dataloader = load_data('Data', batch_size, depth=True)
    print("Evaluating Training dataset with the best model")
    eval_train = evaluation(train_dataloader, model)
    print("Evaluating Testing dataset with the best model")
    results_test = evaluation(test_dataloader, model)
    print('END: Test acc: %0.2f, Train acc: %0.2f' % (results_test, eval_train))

    torch.save(best_model, ("models/model_angloss" + str(round(results_test, 2)) + "_batch" + str(batch_size) +
                        "_e" + str(max_epochs) + "_lr" + str(round(lr, 6)) + ".pth"))