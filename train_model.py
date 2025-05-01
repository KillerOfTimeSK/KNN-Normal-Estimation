import torch
import torch.nn as nn

def TrainModel(model, dataLoader, epochs=10, use_gpu=True):
    if use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        model = model.to('cuda')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for i, (rgb, normal) in enumerate(dataLoader):
            if use_gpu:
                rgb = rgb.to('cuda')
                normal = normal.to('cuda')
            
            optimizer.zero_grad()
            output = model(rgb)
            loss = criterion(output, normal)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataLoader)}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    
    # Save the final model
    torch.save(model.state_dict(), 'model_final.pth')
    # Save the model architecture
    torch.save(model, 'model_architecture.pth')
    return model

def LoadModel(model, model_path, use_gpu=True):
    model.load_state_dict(torch.load(model_path))
    if use_gpu: model = model.to('cuda')
    return model