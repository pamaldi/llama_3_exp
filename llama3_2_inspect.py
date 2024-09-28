import torch

model_path = 'Llama3.2-3B\\consolidated.00.pth'
try:

    state_dict = torch.load(model_path, map_location='cpu')
except Exception as e:
    print(f"Error in loading file: {e}")
    exit(1)

if isinstance(state_dict, dict):
    print("Elements in dict:\n")
    for key, value in state_dict.items():
        print(f"Element: {key}")
        print(f"Values:\n{value}\n")
        weights = state_dict[key]
        weights_float32 = weights.float()
        try:
            weights_np = weights_float32.numpy()
        except TypeError as te:
            print(f"Error in conversion: {te}")
            exit(1)

        print(f"Value (float32):\n{weights_np}\n")
        print(f"Dim weights: {weights.shape}")
        print(f"Dim weights (float32): {weights_float32.shape}")
        print(f"Dim weights (NumPy): {weights_np.shape}")
        print("\n")
else:
    print("Does not contains a dictionary")
    exit(1)

print("\nTotal parameters:")
total_params = 0
curr = 0
for key, value in state_dict.items():
    print(f"Parameter: {key}, Curr: {curr}   Dimension: {value.shape}")
    curr=curr+1
    total_params += value.numel()

print(f"\ntotal parameters: {total_params}")
