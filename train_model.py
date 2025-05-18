import torch
import torch.nn as nn
import torch.nn.functional as F
from model_unet import AngularLoss, CombinedLoss, UNet
import sys, os
import matplotlib.pyplot as plt

def visualize_predictions(image_ids, dataset, model, num_images=3, use_gpu=True, store_dir=None):
    if num_images > len(image_ids): num_images = len(image_ids)
    image_ids = image_ids[:num_images]

    if use_gpu:
        if isinstance(model, UNet): model.ToDevice('cuda')
        else: model = model.cuda()
    
    predictions = []; images = []; normals = []
    model.eval()
    for id in image_ids:
        id = id % len(dataset)
        _, rgbT, tensor, normal = dataset.GetImage(id)
        images.append(rgbT)
        normals.append((normal.permute(1, 2, 0) + 1) / 2)
        if use_gpu: tensor = tensor.to('cuda').unsqueeze(0)
        else: tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            prediction = model(tensor)
            prediction = prediction.cpu().squeeze(0)
            prediction = prediction.permute(1, 2, 0).numpy()
            prediction = (prediction + 1) / 2
            predictions.append(prediction)

    if store_dir is not None and not os.path.exists(store_dir): os.makedirs(store_dir)

    for i in range(len(images)):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title('Input Image')

        if predictions[i].shape != normals[i].shape: 
            print(f"Output shape: {predictions[i].shape}, Normal shape: {normals[i].shape}")
            raise ValueError("Output and predicted shapes do not match.")

        plt.subplot(1, 3, 2)
        plt.imshow(normals[i])
        plt.title('Ground Truth Normal')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i])
        plt.title('Predicted Normal')
        #print(f"Shapes: RGB input:{images[i].size}, GT: {normals[i].shape}, predicted: {predictions[i].shape}")

        #plt.show()
        if store_dir is not None: plt.savefig(os.path.join(store_dir, f'predictions_{i}.png'))
        else: plt.show()
        plt.close()
        plt.clf()

def TrainModel(model, name, dataLoader, file, epochs=10, train_dataset=None, use_gpu=True, LR=1e-3, lrShrink=0.9, maxLR=5e-2, minLR=1e-15, criterion=None, finisherLR=0.2, dropout=0.0):
    if use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        # if isinstance(model, UNet): model.ToDevice('cuda')
        # else: model = model.cuda()
        model.to('cuda')
    else: model.to('cpu')
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': LR},
    #     {'params': model.finisher.parameters(), 'lr': LR * 0.2},
    # ], filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    if finisherLR == 1:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    else:
        finisherParams = set(model.finisher.parameters())
        params = [{'params': filter(lambda p: p.requires_grad and p not in finisherParams, model.parameters()), 'lr': LR}]
        params.append({'params': model.finisher.parameters(), 'lr': LR * finisherLR})
        optimizer = torch.optim.Adam(params, dropout=dropout)
    if criterion is None: criterion = nn.MSELoss()
    #criterion = CombinedLoss()
    #criterion = AngularLoss()
    #criterion = nn.MSELoss()

    image_ids = []
    for i in range(6): image_ids.append(15+20*i)
    
    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for i, (rgb, normal) in enumerate(dataLoader):
            PrintInfo = epoch == 0 and (i == 0 or i == 5 or i == 6 or i == 7 or i == 8 or i == 49 or i == 99)
            #PrintInfo = False
            model.PrintSizes = PrintInfo
            PrintSimple = i % 10 == 0 or (epoch == 0 and (i == 1 or i == 2 or i == 3 or i == 4 or i == 9 or i == 14 or i == 24))
            #PrintInfo = True
            #PrintSimple = True

            IterName = f'B{i+1} E{epoch+1}'
            if isinstance(model, UNet): model.Profiler(f'{IterName}', rgb.shape, PrintInfo)
            if use_gpu:
                rgb = rgb.to('cuda')
                normal = normal.to('cuda')
            
            optimizer.zero_grad()
            output = model(rgb)
            if isinstance(model, UNet): model.Profiler(f'{IterName} (forward)', output.shape, PrintInfo)
            if epoch == 0 and i == 0:
                file(f'Output type: {output.dtype}')
            if output.shape != normal.shape: 
                print(f"Output shape: {output.shape}, Normal shape: {normal.shape}")
                raise ValueError("Output and predicted shapes do not match.")
            loss = criterion(output, normal)
            loss_sum += loss.item()
            if isinstance(model, UNet): model.Profiler(f'Loss {IterName}', loss.shape, PrintInfo)
            loss.backward()
            if isinstance(model, UNet):
                if PrintInfo:
                    file('_' * 50)
                    model.PrintGrads()
                    file('^' * 50)
            if isinstance(model, UNet): model.Profiler(f'{IterName} (backward)', loss.shape, PrintInfo)
            optimizer.step()
            if isinstance(model, UNet): model.Profiler(f'Optimizer {IterName}', loss.shape, PrintInfo)
            
            learnInfo = f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataLoader)}], Loss: {loss.item():.4f}, Average running Loss: {loss_sum/(i+1):.4f}'
            if PrintSimple: file.Important(learnInfo)
            else: file(learnInfo)
        torch.save(model.state_dict(), f'{name}_epoch_{epoch+1}.pth')
        file.Important(f'Epoch [{epoch+1}/{epochs}] finished, Average Loss: {loss_sum/len(dataLoader):.4f}, Used LR={optimizer.param_groups[0]["lr"]:.2e}, Shrink={lrShrink:.2f}')
        file.flush()
        optimizer.param_groups[0]['lr'] *= lrShrink
        if optimizer.param_groups[0]['lr'] <= minLR:
            lrShrink = 1 / lrShrink
            print(f"Learning rate too low ({optimizer.param_groups[0]['lr']:.2e}), will grow back by {lrShrink:.2f}")
            optimizer.param_groups[0]['lr'] = minLR * lrShrink
            print(f"Learning rate set to {optimizer.param_groups[0]['lr']:.2e}")
        if optimizer.param_groups[0]['lr'] >= maxLR:
            lrShrink = 1 / lrShrink
            print(f"Learning rate too high ({optimizer.param_groups[0]['lr']:.2e}), will shrink back by {lrShrink:.2f}")
            optimizer.param_groups[0]['lr'] = maxLR * lrShrink
            print(f"Learning rate set to {optimizer.param_groups[0]['lr']:.2e}")

        visualize_predictions(image_ids, train_dataset, model, 1)
    
    if use_gpu:
        torch.cuda.empty_cache()

    # Save the final model
    torch.save(model.state_dict(), name + '_final.pth')
    # Save the model architecture
    torch.save(model, name + '_architecture.pth')
    return model

def LoadModel(model, model_path, use_gpu=True, version='_final.pth'):
    model.load_state_dict(torch.load(model_path + version))
    if use_gpu: model = model.to('cuda')
    return model

def ValidateModel(model, dataLoader, use_gpu=True):
    if use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        if isinstance(model, UNet): model.ToDevice('cuda')
        else: model = model.cuda()
    else:
        if isinstance(model, UNet): model.ToDevice('cpu')
        else: model = model.cpu()
    model.eval()
    with torch.no_grad():
        AngLoss = AngularLoss()
        loss_sum = 0
        for i, (rgb, normal) in enumerate(dataLoader):
            if use_gpu:
                rgb = rgb.to('cuda')
                normal = normal.to('cuda')
            output = model(rgb)
            if output.shape != normal.shape: 
                print(f"Output shape: {output.shape}, Normal shape: {normal.shape}")
                raise ValueError("Output and predicted shapes do not match.")
            loss = AngLoss(output, normal)
            loss_sum += loss.item()
            if i % 50 == 9: print(f'Validation Step [{i+1}/{len(dataLoader)}], Loss: {loss.item():.4f}')
        print(f'Validation finished Average Loss: {loss_sum/len(dataLoader):.4f}')


# Epoch [1/4], Step [3/263], Loss: 0.4121
# Event name: Batch 4 of epoch 1 - Time: 49s, Memory: 1.3GB, Shape: torch.Size([32, 3, 768, 1024])
# Event name: Output 4 of epoch 1 (forward pass) - Time: 49s, Memory: 5.0GB, Shape: torch.Size([32, 3, 16, 16])
# Event name: Loss 4 of epoch 1 - Time: 49s, Memory: 4.7GB, Shape: torch.Size([])
# Event name: Loss 4 of epoch 1 (backward pass) - Time: 49s, Memory: 1.5GB, Shape: torch.Size([])
# Event name: Optimizer step 4 of epoch 1 - Time: 49s, Memory: 1.5GB, Shape: torch.Size([])
# Epoch [1/4], Step [4/263], Loss: 0.3570
# Event name: Batch 5 of epoch 1 - Time: 53s, Memory: 1.3GB, Shape: torch.Size([32, 3, 768, 1024])
# Event name: Output 5 of epoch 1 (forward pass) - Time: 53s, Memory: 5.0GB, Shape: torch.Size([32, 3, 16, 16])
# Event name: Loss 5 of epoch 1 - Time: 53s, Memory: 4.7GB, Shape: torch.Size([])
# Event name: Loss 5 of epoch 1 (backward pass) - Time: 53s, Memory: 1.5GB, Shape: torch.Size([])
# Event name: Optimizer step 5 of epoch 1 - Time: 53s, Memory: 1.5GB, Shape: torch.Size([])
# Epoch [1/4], Step [5/263], Loss: 0.3712
# Event name: Batch 10 of epoch 1 - Time: 69s, Memory: 1.3GB, Shape: torch.Size([32, 3, 768, 1024])
# Event name: Output 10 of epoch 1 (forward pass) - Time: 73s, Memory: 5.0GB, Shape: torch.Size([32, 3, 16, 16])
# Event name: Loss 10 of epoch 1 - Time: 73s, Memory: 4.7GB, Shape: torch.Size([])
# Event name: Loss 10 of epoch 1 (backward pass) - Time: 73s, Memory: 1.5GB, Shape: torch.Size([])
# Event name: Optimizer step 10 of epoch 1 - Time: 73s, Memory: 1.5GB, Shape: torch.Size([])
# Epoch [1/4], Step [10/263], Loss: 0.4696