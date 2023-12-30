import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from Param_Our import *
import CMT
import model_structure
from tqdm import tqdm

def get_model_structure(name):
    model = model_structure.get_model(name)
    model.summary()
    return model

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    mape = []
    for i in range(len(y_true)):
        if (y_true[i] == 0):
            mape.append(abs(y_pred[i]) * 100)
        else:
            mape.append((np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100))
    return np.mean(mape)

def MAPE2(y_true, y_pred):
    y_true = np.sum(y_true, axis=-1)
    y_pred = np.sum(y_pred, axis=-1)
    mape = []
    for i in range(np.shape(y_true)[0]):
        mape.append((np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100))
    return np.mean(mape)

def data_slider(data, type_of_data, previous_time_input=TIMESTEP_OUR, prediction_of_time=Prediction_TIMESTEP):
    data = np.array(data)
    data_array = []
    for i in range(np.shape(data)[0] - (previous_time_input + prediction_of_time) + 1):
        temp = data[i:i + previous_time_input]
        data_array.append(temp)
    data_array = np.array(data_array)
    return data_array

def main(train=True):
    # Get the directory path of the current script
    base_directory = os.path.dirname(__file__)
    directory_path = os.path.join(base_directory)  # Example directory name
    data_path = os.path.join(base_directory)  # Example data folder name
    model_path = os.path.join(base_directory, 'checkpoint.pth')
    os.chdir(directory_path)
    previous_time_input = TIMESTEP_OUR  # Input time slots (4-hour input)
    prediction_of_time = Prediction_TIMESTEP  # Output (1-hour prediction)
    os.makedirs(os.path.join(base_directory, 'Figures'), exist_ok=True)
    region_path = os.path.join(data_path, 'flowioK_TaxiNYC_20160101_20160228_60min.npy')
    temporal_path = os.path.join(data_path, 'Temporal_Master_Grid_NYC30x30(60_min_resolution).npy')
    model_name = "CMT"
    dataset = 'Careem'

    region = np.load(region_path)
    temporal = np.load(temporal_path)

    #Scaling the region data
    scaled_data_region = np.empty_like(region)
    scalers_region = {}
    # Loop over the second dimension (channels)
    for i in range(region.shape[1]):
        scaler = StandardScaler()
        # Reshape the data to 2D - [samples * height * width, 1]
        flattened_region_data = region[:, i, :, :].reshape(-1, 1)
        scaled_flattened_data = scaler.fit_transform(flattened_region_data)
        # Reshape back to original dimensions
        scaled_data_region[:, i, :, :] = scaled_flattened_data.reshape(region.shape[0], region.shape[2], region.shape[3])
        scalers_region[i] = scaler  # Store the scaler for each channel

    # Scaling temporal data
    scaled_data_temporal = np.empty_like(temporal)
    scalers_temporal = {}
    # Loop over the second dimension (channels)
    for i in range(temporal.shape[1]):
        scaler = StandardScaler()
        # Reshape the data to 2D - [samples * height * width, 1]
        flattened_temporal_data = temporal[:, i, :, :].reshape(-1, 1)
        scaled_flattened_data = scaler.fit_transform(flattened_temporal_data)
        # Reshape back to original dimensions
        scaled_data_temporal[:, i, :, :] = scaled_flattened_data.reshape(temporal.shape[0], temporal.shape[2], temporal.shape[3])
        scalers_temporal[i] = scaler  # Store the scaler for each channel

    temporal = np.array(scaled_data_temporal)
    region = np.transpose(scaled_data_region, (0, 3, 1, 2))
    # temporal = np.array(temporal)
    # region = np.transpose(region, (0, 3, 1, 2))
    print("TEMPORAL FILE SHAPE:", temporal.shape)
    print("SPATIAL FILE SHAPE:", region.shape)

    region_data = data_slider(region, "Spatial")
    region_data = region_data.reshape((region_data.shape[0], previous_time_input, local_image_size_x, local_image_size_y))
    temporal_data = data_slider(temporal, "Temporal")

    y_true = []
    for i in range(np.shape(region)[0] - (previous_time_input + prediction_of_time) + 1):
        temp = region[i + previous_time_input:i + (previous_time_input + prediction_of_time)]
        y_true.append(temp)
    y_true = np.array(y_true)

    trainx_1, valx_1, trainx_2, valx_2, ytrain, val_y = train_test_split(region_data, temporal_data, y_true,
                                                                         train_size=TRAIN_RATIO, random_state=42, shuffle=True)

    print("-"*100)
    print("Training Data: ")
    print("Regional data training shape:", trainx_1.shape)
    print("Temporal data training shape:", trainx_2.shape)
    print("True values Training shape:  ", ytrain.shape)
    print("-"*100)

    print("Val Data: ")
    print("Regional data Validation shape:", valx_1.shape)
    print("Temporal data Validation shape:", valx_2.shape)
    print("True values Validation shape:  ", val_y.shape)
    print("-"*100)

    model = CMT.CMT_Ti()
    optimizer = optim.Adam(model.parameters(), lr=LEARN, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Convert data to PyTorch tensors
    trainx_1_tensor = torch.tensor(trainx_1, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    valx_1_tensor = torch.tensor(valx_1, dtype=torch.float32)
    yval_tensor = torch.tensor(val_y, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = TensorDataset(trainx_1_tensor, ytrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    
    checkpoint_interval = 30  # Save state of model after every 30 epochs
    loss_dict ={}
    if train:
        progress_bar = tqdm(range(EPOCH), ncols=100, position=0)
        for epoch in progress_bar:
            model.train()
            total_loss = 0.0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step()
            # if epoch == 0 or (epoch+1)%100 == 0:
            loss_dict[epoch+1] = avg_loss
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch: {epoch+1}/{EPOCH} -> MSE Loss: {avg_loss:.4f}, LR: {current_lr}")

            # if epoch % checkpoint_interval == 0:
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': avg_loss,
            #         # 'output': output
            #     }, model_path)
        print("-"*100)
        print("Training is done")
        print("-"*100)

    # Evaluate on validation data
    model.eval()
    with torch.no_grad():
        # predictions = model(trainx_1_tensor)
        predictions = model(valx_1_tensor)
    predictions = predictions.numpy()

    print(f"Val y shape : {val_y.shape}")
    print(f"Pred y shape : {predictions.shape}")

    # reshaped_true = np.sum(ytrain, axis=1).reshape(ytrain.shape[0], local_image_size_x * local_image_size_y)
    reshaped_true = np.sum(val_y, axis=1).reshape(val_y.shape[0], local_image_size_x * local_image_size_y)
    reshaped_pred = np.sum(predictions, axis=1).reshape(predictions.shape[0], local_image_size_x * local_image_size_y)

    plt.figure(figsize=(25, 8))
    plt.title('ACTUAL VS PREDICTED DEMAND Validation', fontsize=24)
    plt.plot(np.sum(reshaped_true, axis=1))
    plt.plot(np.sum(reshaped_pred, axis=1))
    plt.legend(['Actual', 'Predicted'], fontsize=15)
    plt.ylabel('Total Demand', fontsize=20)
    plt.xlabel('Time (Hour)', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'ACTUAL VS PREDICTED DEMAND OF LAHORE VALIDATION')
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.title('TRAINING LOSS', fontsize=24)
    plt.plot(list(loss_dict.keys()), list(loss_dict.values()), label='Train Loss')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'TRAINING LOSS CURVE')
    plt.close()

    print("Validation MAE:", mae(reshaped_true, reshaped_pred))
    print("Validation RMSE:", math.sqrt(mse(reshaped_true, reshaped_pred)))
    print("Validation MAPE:", MAPE(reshaped_true, reshaped_pred))
    print("Validation MAPE2:", MAPE2(reshaped_true, reshaped_pred))

if __name__ == "__main__":
    main()