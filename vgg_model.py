from numpy.f2py.auxfuncs import throw_error
from torchvision.models import vgg16, vgg16_bn
import torch.nn as nn
from dataset import load_data
import torch
import matplotlib.pyplot as plt
import copy
from main import angular_error
import numpy as np


class VGGNormal(nn.Module):
    def __init__(self):
        super(VGGNormal, self).__init__()
        # Use the VGG16 feature extractor
        self.encoder = vgg16(weights='DEFAULT').features
        # Initialize the decoder
        self.decoder = nn.Sequential(
            # First deconvolution layer: Upsample to (14, 14, 256)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Second deconvolution layer: Upsample to (28, 28, 128)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Third deconvolution layer: Upsample to (56, 56, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Fourth deconvolution layer: Upsample to (112, 112, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Final deconvolution layer: Upsample to (224, 224, 3) - original image size with 3 channels
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid to get pixel values in the range [0, 1]
        )

    def forward(self, x):
        # Pass the image through the VGG16 feature extractor
        features = self.encoder(x)
        # Reconstruct the image using the decoder
        reconstructed_image = self.decoder(features)
        return reconstructed_image


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
            inputs, normals = inputs.to(device), normals.to(device)
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
    # TODO tidy up the code and fragment it into functions or separate file
    # untrained angular error for val dataset = 29.03
    # untrained angular error for train dataset = 120.21

    for batch_size in [64, 256]:
        model = VGGNormal().cuda()
        train_dataloader, test_dataloader = load_data('Data', batch_size)
        loss_epoch_arr = []
        best_model = model.state_dict()

        max_epochs = 4
        min_loss = 1000000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        n_iters = len(train_dataloader)
        model.train()
        for epoch in range(max_epochs):

            for i, data in enumerate(train_dataloader, 0):

                inputs= data['image']
                normals = data['normal']
                inputs, normals = inputs.to(device), normals.to(device)

                opt.zero_grad()

                outputs = model(inputs)
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
            print("Evaluating Test dataset at the end of epoch %d" % (epoch + 1))
            eval_test = evaluation(test_dataloader, model)
            print("Evaluating Training dataset at the end of epoch %d" % (epoch + 1))
            eval_train = evaluation(train_dataloader, model)
            print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
                epoch + 1, max_epochs,
                eval_test, eval_train))

        model.load_state_dict(best_model)
        model.eval()
        print("Evaluating Testing dataset with the best model")
        results_test = evaluation(test_dataloader, model)
        print('END: Test acc: %0.2f' % results_test)
        # change max_epochs value to > 1
        #plt.plot(loss_epoch_arr)
        #plt.show()

        torch.save(best_model, ("models/model_angloss" + str(round(results_test, 2)) + "_batch" + str(batch_size) +
                                "_e" + str(max_epochs) + ".pth"))


    model.eval()
    for sample in test_dataloader:
        img = sample['image'].squeeze(0).permute(1, 2, 0).numpy()
        img_tensor = sample['image'].cuda()
        gt_image = sample['normal'].squeeze(0).permute(1, 2, 0).numpy()
        with torch.no_grad():
            reconstructed_image = model(img_tensor)
            reconstructed_image = reconstructed_image.cpu().squeeze(0).permute(1, 2, 0).numpy()  # Convert to numpy array


            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.subplot(1, 3, 2)
            plt.imshow(reconstructed_image)
            plt.title('Reconstructed Image')
            plt.subplot(1, 3, 3)
            plt.imshow(gt_image)
            plt.title('Ground Truth Image')
            plt.show()
        exit(0)