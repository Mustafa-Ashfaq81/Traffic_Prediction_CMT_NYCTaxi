import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error  as mae
from sklearn.preprocessing import StandardScaler
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
        if(y_true[i] == 0):
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
    base_directory = os.path.dirname(__file__)  # Get the directory path of the current script
    
    directory_path = os.path.join(base_directory)  # Example directory name
    data_path = os.path.join(base_directory)  # Example data folder name
    model_path = os.path.join(base_directory, 'checkpoint.pth')  # Example model folder name

    os.chdir(directory_path)
    previous_time_input = TIMESTEP_OUR  # Input time slots (4-hour input)
    prediction_of_time = Prediction_TIMESTEP  # Output (1-hour prediction)
    m_name = 'Demand'
    model_name = "CMT"
    dataset = 'Careem'

    os.makedirs(os.path.join(base_directory, 'Figures'), exist_ok=True)

    region_path = os.path.join(data_path, 'flowioK_TaxiNYC_20160101_20160228_60min.npy')
    temporal_path = os.path.join(data_path, 'Temporal_Master_Grid_NYC30x30(60_min_resolution).npy')

    region = np.load(region_path)
    temporal = np.load(temporal_path)

    temporal = np.array(temporal)
    region = np.transpose(region, (0, 3, 1, 2))
    print("TEMPORAL FILE SHAPE:",temporal.shape)
    print("SPATIAL FILE SHAPE:",region.shape)

    region_data = data_slider(region, "Spatial")
    region_data = region_data.reshape((region_data.shape[0], previous_time_input, local_image_size_x, local_image_size_y))

    temporal_data = data_slider(temporal, "Temporal")

    y_true = []
    for i in range(np.shape(region)[0] - (previous_time_input + prediction_of_time) + 1):
        temp = region[i + previous_time_input:i + (previous_time_input + prediction_of_time)]
        y_true.append(temp)
    y_true = np.array(y_true)

    trainx_1, valx_1, trainx_2, valx_2, ytrain, val_y = train_test_split(region_data, temporal_data, y_true,
                                                                        train_size=0.8, random_state=1, shuffle=True)

    print("-"*100)
    print("Training Data: ")
    print("Regional data training shape:",trainx_1.shape)
    print("Temporal data training shape:",trainx_2.shape)
    print("True values Training shape:  ",ytrain.shape)
    print("-"*100)

    print("Val Data: ")
    print("Regional data Validation shape:",valx_1.shape)
    print("Temporal data Validation shape:",valx_2.shape)
    print("True values Validation shape:  ",val_y.shape)
    print("-"*100)
    
    
    model = CMT.CMT_S()
    # optimizer = torch.optim.AdamW(model.parameters(), lr = LEARN, weight_decay = WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARN)

    # Convert data to PyTorch tensors
    trainx_1_tensor = torch.tensor(trainx_1, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    # Add scaling part here if needed
    # Create DataLoader for batching
    train_dataset = TensorDataset(trainx_1_tensor, ytrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    # loss_list = []
    checkpoint_interval = 1 #Save state of model after every epoch
    if train:
        progress_bar = tqdm(range(EPOCH), ncols=100, position=0)
        for epoch in progress_bar:
            model.train()
            total_loss = 0.0

            for inputs, targets in train_loader:
                # print(inputs.shape)
                optimizer.zero_grad()
                # print(f"\ninput tensor: \n{inputs}\n")

                outputs = model(inputs)

                # print(f"output tensor: \n{outputs}\n")
                # print(f"Target: \n{targets}")

                # Calculating the loss
                # squared_diff = (outputs - targets)**2
                # # Calculate the mean along all dimensions except the batch dimension
                # loss = torch.mean(squared_diff)
                loss = torch.nn.functional.mse_loss(outputs, targets)**0.5
                # loss = mae(outputs, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # if checkpoint_interval == 1:
                #     break
            
            avg_loss = total_loss / len(train_loader)

            # if epoch == 0 or (epoch+1)%100 == 0:
            print(f"\nEpoch: {epoch+1}/{EPOCH} -> RMSE Loss: {avg_loss:.4f}")

            if epoch % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    # 'output': output
                }, model_path)
        print("-"*100)
        print("Training is done")
        print("-"*100)

    # Evaluate on training data
    model.eval()
    with torch.no_grad():
        predictions = model(trainx_1_tensor)
    predictions = predictions.numpy()

    reshaped_true = np.sum(ytrain, axis=1).reshape(ytrain.shape[0], local_image_size_x * local_image_size_y)
    reshaped_pred = np.sum(predictions, axis=1).reshape(predictions.shape[0], local_image_size_x * local_image_size_y)

    plt.figure(figsize=(25, 5))
    plt.title('ACTUAL VS PREDICTED DEMAND TRAINING', fontsize=24)
    plt.plot(np.sum(reshaped_true, axis=1))
    plt.plot(np.sum(reshaped_pred, axis=1))
    plt.legend(['Actual', 'Predicted'], fontsize=15)
    plt.ylabel('Total Demand', fontsize=20)
    plt.xlabel('Time (Hour)', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'ACTUAL VS PREDICTED DEMAND OF LAHORE TRAINING')
    plt.close()

    print("Training MAE:", mae(reshaped_true, reshaped_pred))
    print("Training RMSE:", math.sqrt(mse(reshaped_true, reshaped_pred)))
    print("Training MAPE:", MAPE(reshaped_true, reshaped_pred))
    print("Training MAPE2:", MAPE2(reshaped_true, reshaped_pred))

if __name__ == "__main__":
    main()

# for scaling input data

# # Flatten the 4D input to 2D, where each row represents a data sample
# trainx_1_flattened = trainx_1_tensor.view(trainx_1_tensor.size(0), -1)
# valx_1_flattened = valx_1_tensor.view(valx_1_tensor.size(0), -1)

# # Create the StandardScaler and fit/transform it on the training data
# scaler = StandardScaler()
# trainx_1_scaled = scaler.fit_transform(trainx_1_flattened)

# # You can transform the validation data using the same scaler
# valx_1_scaled = scaler.transform(valx_1_flattened)

# # You can reshape them back to the original shape if needed
# trainx_1_scaled = trainx_1_scaled.reshape(trainx_1_tensor.shape)
# valx_1_scaled = valx_1_scaled.reshape(valx_1_tensor.shape)
# # Reshape ytrain_tensor to have the same shape as trainx_1_scaled
# # Ensure ytrain_tensor is converted to a numpy array
# ytrain_numpy = ytrain_tensor.numpy()

# # Reshape ytrain_numpy to have the same shape as trainx_1_scaled
# ytrain_numpy = ytrain_numpy.squeeze(1)  # Remove extra dimensions

# # Convert ytrain_numpy back to a PyTorch tensor
# ytrain_tensor = torch.tensor(ytrain_numpy, dtype=torch.float32)
# # Ensure trainx_1_scaled is converted to a PyTorch tensor
# trainx_1_scaled_tensor = torch.tensor(trainx_1_scaled, dtype=torch.float32)

# print("trainx_1_scaled shape:", trainx_1_scaled_tensor.shape)
# print("ytrain_tensor shape:", ytrain_tensor.shape)

# # Create DataLoader for training data
# train_dataset = TensorDataset(trainx_1_scaled_tensor, ytrain_tensor)
# train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)