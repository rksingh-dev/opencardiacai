import torch
import shutil
import os

print("Zipping the model folder back into a .pt file...")
shutil.make_archive('temp_model', 'zip', 'model')
os.rename('temp_model.zip', 'temp_model.pt')

try:
    print("Attempting to load with torch.jit.load...")
    model = torch.jit.load('temp_model.pt')
    print("Successfully loaded as TorchScript!")
    print(model)
except Exception as e1:
    print("Failed as TorchScript:", e1)
    try:
        print("Attempting to load with standard torch.load...")
        model = torch.load('temp_model.pt', map_location='cpu')
        print("Successfully loaded with torch.load!")
        print(type(model))
    except Exception as e2:
        print("Failed as standard load:", e2)
