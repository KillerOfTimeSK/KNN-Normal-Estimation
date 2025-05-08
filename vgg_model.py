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
        self.vgg = vgg16(weights='DEFAULT').features
        self.layer_ids = [4, 9, 16, 23]
        # Initialize the decoder
        self.decoder0 = nn.Sequential(
            # Initial deconvolution layer: Upsample to (14, 14, 512) - same as 23rd layer of vgg16
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            # First deconvolution layer: Upsample to (28, 28, 256) - same as 16th layer of vgg16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            # Second deconvolution layer: Upsample to (56, 56, 128) - same as 9th layer of vgg16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            # Third deconvolution layer: Upsample to (112, 112, 64) same as 4th layer of vgg16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            # Fourth deconvolution layer: Upsample to (224, 224, 32) - same as input image HxW
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder5 = nn.Sequential(
            # Final deconvolution layer: Upsample to (224, 224, 3) - same dimensions and channel number as input
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )
        self.final = nn.Tanh()

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                outputs[f'layer_{i}'] = x
        # Reconstruct the image using the decoder
        x = self.decoder0(x) + outputs['layer_23']
        x = self.decoder1(x) + outputs['layer_16']
        x = self.decoder2(x) + outputs['layer_9']
        x = self.decoder3(x) + outputs['layer_4']
        x = self.decoder4(x)
        x = self.decoder5(x)
        # Tanh for output in range <-1,1>
        return self.final(x)


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_size in [32, 16, 2, 64, 128]:
        model = VGGNormal().cuda()
        train_dataloader, test_dataloader = load_data('Data', batch_size)
        loss_epoch_arr = []
        best_model = model.state_dict()

        max_epochs = 4
        min_loss = 1000000


        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (epoch + 1, max_epochs, eval_test, eval_train))

        model.load_state_dict(best_model)
        model.eval()
        print("Evaluating Testing dataset with the best model")
        results_test = evaluation(test_dataloader, model)
        print('END: Test acc: %0.2f' % results_test)
        #if max_epochs > 1:
        #    plt.plot(loss_epoch_arr)
        #    plt.show()

        torch.save(best_model, ("models/model_angloss" + str(round(results_test, 2)) + "_batch" + str(batch_size) +
                                "_e" + str(max_epochs) + ".pth"))

    #model = VGGNormal().cuda()
    #model.load_state_dict(torch.load("models/", weights_only=True))
    #train_dataloader, test_dataloader = load_data('Data', 1)
    model.eval()
    i = 0
    for sample in test_dataloader:
        img = sample['image'].squeeze(0).permute(1, 2, 0).numpy()
        img_tensor = sample['image'].cuda()
        gt_image = sample['normal'].squeeze(0).permute(1, 2, 0).numpy()
        with torch.no_grad():
            reconstructed_image = model(img_tensor)
            reconstructed_image = reconstructed_image.cpu().squeeze(0).permute(1, 2, 0).numpy()  # Convert to numpy array

            ang = np.mean(angular_error(gt_image, reconstructed_image))
            if ang > 30:
                continue
            print(ang)
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
        i += 1
        if i >= 4:
            exit(0)
