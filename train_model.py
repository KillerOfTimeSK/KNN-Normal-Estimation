import torch
import torch.nn as nn
import torch.nn.functional as F
from model_unet import AngularLoss

def TrainModel(model, name, dataLoader, epochs=10, use_gpu=True):
    if use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        model.MoveToGPU()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = AngularLoss()
    
    for epoch in range(epochs):
        for i, (rgb, normal) in enumerate(dataLoader):
            PrintInfo = epoch == 0 and (i == 0 or i == 1 or i == 2 or i == 49 or i == 99)
            PrintSimple = i % 100 == 0 or (epoch == 0 and (i == 1 or i == 2 or i == 3 or i == 4 or i == 9 or i == 14 or i == 24 or i == 49 or i == 74))
            model.PrintSizes = epoch == 0 and i == 0

            IterName = f'B{i+1} E{epoch+1}'
            model.Profiler(f'{IterName}', rgb.shape, PrintInfo)
            if use_gpu:
                rgb = rgb.to('cuda')
                normal = normal.to('cuda')
            
            optimizer.zero_grad()
            output = model(rgb)
            model.Profiler(f'{IterName} (forward)', output.shape, PrintInfo)
            if epoch == 0 and i == 0:
                print(f'Output type: {output.dtype}')
            size = output.shape[2:]
            normal = F.interpolate(normal, size=size, mode='bilinear', align_corners=True)
            loss = criterion(output, normal)
            model.Profiler(f'Loss {IterName}', loss.shape, PrintInfo)
            loss.backward()
            model.Profiler(f'{IterName} (backward)', loss.shape, PrintInfo)
            optimizer.step()
            model.Profiler(f'Optimizer {IterName}', loss.shape, PrintInfo)
            
            if PrintSimple: print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataLoader)}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), f'{name}_epoch_{epoch+1}.pth')
    
    if use_gpu:
        torch.cuda.empty_cache()

    # Save the final model
    torch.save(model.state_dict(), name + '_final.pth')
    # Save the model architecture
    torch.save(model, name + '_architecture.pth')
    return model

def LoadModel(model, model_path, use_gpu=True):
    model.load_state_dict(torch.load(model_path + '_final.pth'))
    if use_gpu: model = model.to('cuda')
    return model



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