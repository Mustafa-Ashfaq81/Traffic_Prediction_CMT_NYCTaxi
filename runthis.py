import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from Param_Our import *
import CMT

def MAPE(y_true, y_pred): # the mean absolute percentage errors
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    mape = []
    for i in range(len(y_true)):
        if (y_true[i] == 0):
            mape.append(abs(y_pred[i]) * 100)
        else:
            mape.append((np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100))
    return np.mean(mape)

def MAPE2(y_true, y_pred): # the mean absolute percentage errors with time series(sequential data)
    y_true = np.sum(y_true, axis=-1)
    y_pred = np.sum(y_pred, axis=-1)
    mape = []
    for i in range(np.shape(y_true)[0]):
        # mape.append((np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100))
        mape.append((np.abs((y_true[i] - y_pred[i]) / (y_true[i] + 1e-8)) * 100))
    return np.mean(mape)

def data_slider(data, previous_time_input = PREVIOUS_TIMESTEP, prediction_of_time = PREDICTION_TIMESTEP):
    data = np.array(data)
    data_array = []
    for i in range(np.shape(data)[0] - (previous_time_input + prediction_of_time) + 1):
        temp = data[i:i + previous_time_input]
        data_array.append(temp)
    data_array = np.array(data_array)
    return data_array

def main(train=False):
    # Get the directory path of the current script
    base_directory = os.path.dirname(__file__)
    directory_path = os.path.join(base_directory)  # Example directory name
    data_path = os.path.join(base_directory)  # Example data folder name
    model_path = os.path.join(base_directory, 'checkpoint.pth')
    os.chdir(directory_path)
    os.makedirs(os.path.join(base_directory, 'Figures'), exist_ok=True)
    data_path = os.path.join(data_path, 'NYC_Taxi_JuneFinal_30x30.npy')
    model_name = 'CMT'
    dataset = 'NYC_TAXI'

    region = np.load(data_path)
    region = np.transpose(region, (0, 3, 1, 2))

    print("-"*100)
    print("SPATIO-TEMPORAL FILE SHAPE:", region.shape)

    region_data = data_slider(region)
    region_data = region_data.reshape((region_data.shape[0], PREVIOUS_TIMESTEP, local_image_size_x, local_image_size_y))

    y_true = []
    for i in range(np.shape(region)[0] - (PREVIOUS_TIMESTEP + PREDICTION_TIMESTEP) + 1):
        temp = region[i + PREVIOUS_TIMESTEP:i + (PREVIOUS_TIMESTEP + PREDICTION_TIMESTEP)]
        y_true.append(temp)
    y_true = np.array(y_true)

    y_true = y_true.reshape((y_true.shape[0], PREDICTION_TIMESTEP*local_image_size_x*local_image_size_y))

    # Train-Test Split
    trainx = region_data[:int(np.shape(region_data)[0]*TRAIN_RATIO)]
    testx = region_data[int(np.shape(region_data)[0]*TRAIN_RATIO):]
    ytrain = y_true[:int(np.shape(region_data)[0]*TRAIN_RATIO)]
    ytest = y_true[int(np.shape(region_data)[0]*TRAIN_RATIO):]

    trainx, valx, ytrain, yval = train_test_split(trainx, ytrain, train_size=TRAIN_RATIO,
                                                  random_state=42, shuffle=True)

    print("-"*100)
    print("Training Data: ")
    print("Spatio-Temporal data training shape:", trainx.shape)
    print("True values Training shape:  ", ytrain.shape)
    print("-"*100)
    print("Val Data: ")
    print("Spatio-Temporal data Validation shape:", valx.shape)
    print("True values Validation shape:  ", yval.shape)
    print("-"*100)
    print("Test Data: ")
    print("Spatio-Temporal data test shape:",testx.shape)
    print("True values test shape:  ",ytest.shape)
    print("-"*100)

    model = CMT.CMT_Ti()
    optimizer = optim.Adam(model.parameters(), lr=LEARN, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=280, gamma=0.1)

    # Convert data to PyTorch tensors
    trainx_tensor = torch.tensor(trainx, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    valx_tensor = torch.tensor(valx, dtype=torch.float32)
    yval_tensor = torch.tensor(yval, dtype=torch.float32)
    testx_tensor = torch.tensor(testx, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = TensorDataset(trainx_tensor, ytrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_dataset = TensorDataset(valx_tensor, yval_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True)
    
    checkpoint_interval = 100  # Save state of model after every 100 epochs
    train_loss_dict ={}
    val_loss_dict = {}
    if train:
        progress_bar = tqdm(range(EPOCH), ncols=100, position=0)
        for epoch in progress_bar:
            total_train_loss = 0.0
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            scheduler.step()

            train_loss_dict[epoch+1] = avg_train_loss
            current_lr = scheduler.get_last_lr()[0]

            # Validation phase
            val_loss = 0.0
            model.eval()
            with torch.no_grad():  # No gradient computation during validation
                for val_inputs, val_targets in val_loader:
                    val_outputs = model(val_inputs)
                    val_batch_loss = criterion(val_outputs, val_targets)
                    val_loss += val_batch_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_loss_dict[epoch+1] = avg_val_loss

            if epoch == 0 or (epoch+1)%100 == 0:
                print(f"\nEpoch: {epoch+1}/{EPOCH} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr}")

            if (epoch+1) % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_dict,
                    'val_loss' : val_loss_dict
                }, model_path)
        print("-"*100)
        print("Training is done")
        print("-"*100)
    else:
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        train_loss_dict = checkpoint['train_loss']
        val_loss_dict = checkpoint['val_loss']

    # Train & Val Loss
    plt.figure(figsize=(25, 10))
    plt.title('TRAINING & VAL LOSS', fontsize=24)
    plt.plot(list(train_loss_dict.keys()), list(train_loss_dict.values()))
    plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()))
    plt.legend(['Train Loss', 'Val Loss'], fontsize=15)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'TRAINING & VAL LOSS CURVE')
    plt.close()

    # Evaluate on validation, test data
    model.eval()
    with torch.no_grad():
        pred_train = model(trainx_tensor).numpy()
        pred_val = model(valx_tensor).numpy()
        pred_test = model(testx_tensor).numpy()

    train_true = ytrain
    train_pred = pred_train
    val_true = yval
    val_pred = pred_val
    test_true = ytest
    test_pred = pred_test

    # Traning
    plt.figure(figsize=(25, 8))
    plt.title('ACTUAL VS PREDICTED DEMAND TRAINING', fontsize=24)
    plt.plot(np.sum(train_true, axis=1))
    plt.plot(np.sum(train_pred, axis=1))
    plt.legend(['Actual', 'Predicted'], fontsize=15)
    plt.ylabel('Total Demand', fontsize=20)
    plt.xlabel('Time (Hour)', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'ACTUAL VS PREDICTED DEMAND TRAINING')
    plt.close()

    print("Training MAE:",mae(train_true,train_pred))
    print("Training RMSE:",math.sqrt(mse(train_true,train_pred)))
    print("Training MAPE:",MAPE(train_true,train_pred)) #change
    print("Training MAPE2:",MAPE2(train_true,train_pred))
    print("-"*100)

    # Validation
    plt.figure(figsize=(25, 8))
    plt.title('ACTUAL VS PREDICTED DEMAND VALIDATION', fontsize=24)
    plt.plot(np.sum(val_true, axis=1))
    plt.plot(np.sum(val_pred, axis=1))
    plt.legend(['Actual', 'Predicted'], fontsize=15)
    plt.ylabel('Total Demand', fontsize=20)
    plt.xlabel('Time (Hour)', fontsize=20)
    plt.savefig('Figures/' + model_name + '_' + dataset + '_' + 'ACTUAL VS PREDICTED DEMAND VALIDATION')
    plt.close()

    print("Validation MAE:", mae(val_true, val_pred))
    print("Validation RMSE:", math.sqrt(mse(val_true, val_pred)))
    print("Validation MAPE:", MAPE(val_true, val_pred)) #change
    print("Validation MAPE2:", MAPE2(val_true, val_pred))
    print("-"*100)

    # Test
    plt.figure(figsize=(25,5))
    plt.title('ACTUAL VS PREDICTED DEMAND TESTING',fontsize=24)
    plt.plot(np.sum(test_true,axis=1))
    plt.plot(np.sum(test_pred,axis=1))
    plt.legend(['Actual','Predicted'],fontsize=15)
    plt.ylabel('Total Demand',fontsize=20)
    plt.xlabel('Time (Hour)',fontsize=20)
    plt.savefig('Figures/'+model_name+'_'+dataset+'_'+'ACTUAL VS PREDICTED DEMAND TESTING')
    plt.close()

    print("Testing MAE:",mae(test_true,test_pred))
    print("Testing RMSE:",math.sqrt(mse(test_true,test_pred)))
    print("Testing MAPE:",MAPE(test_true,test_pred)) #change
    print("Testing MAPE2:",MAPE2(test_true,test_pred))
    print("-"*100)

    # Overall
    region_data_tensor = torch.tensor(region_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_overall = model(region_data_tensor).numpy()

    x = np.linspace(PREVIOUS_TIMESTEP + 1, PREVIOUS_TIMESTEP + len(pred_overall), len(pred_overall))  #Results start from the 5th hour upto x hours
    
    plt.figure(figsize=(25,5))
    plt.title('ACTUAL VS PREDICTED DEMAND FOR TWO MONTHS',fontsize=24)
    plt.plot(x,np.sum(y_true,axis=1))
    plt.plot(x,np.sum(pred_overall,axis=1))
    plt.legend(['Actual','Predicted'],fontsize=15)
    plt.ylabel('Total Demand',fontsize=20)
    plt.xlabel('Time (Hour)',fontsize=20)
    plt.savefig('Figures/'+model_name+'_'+dataset+'_'+'ACTUAL VS PREDICTED DEMAND OVERALL')
    plt.close()

    print('Overall MAE:',mae(y_true,pred_overall))
    print('Overall RMSE:',math.sqrt(mse(y_true,pred_overall)))
    print('Overall MAPE:',MAPE(y_true,pred_overall)) #change
    print('Overall MAPE2:',MAPE2(y_true,pred_overall))
    print("-"*100)

    # Rolling
    Hour = 168 # 1 week selected for rolling prediction from the end of the data 

    rolling_pred_array = []
    last24_region = region_data_tensor[-Hour-1:-Hour]

    for i in range(Hour):
        last24_region = last24_region.reshape(PREDICTION_TIMESTEP, PREVIOUS_TIMESTEP, local_image_size_x, local_image_size_y)
        model.eval()
        with torch.no_grad():
            pred = model(last24_region)

        rolling_pred_array.append(pred.numpy().squeeze(0))

        pred = pred.reshape(PREDICTION_TIMESTEP, local_image_size_x, local_image_size_y)

        # Remove the oldest hour 
        last23_region = last24_region[:, PREDICTION_TIMESTEP:, :, :]
        # Append the predicted hour
        last24_region = torch.cat((last23_region, pred.unsqueeze(0)), axis=1)

    rolling_true = y_true[-Hour:,:]
    rolling_pred_array = np.array(rolling_pred_array)

    plt.figure(figsize=(25,5))
    plt.title('ACTUAL VS ROLLING PREDICTED DEMAND FOR ONE WEEK',fontsize=24)
    plt.plot(np.sum(rolling_true,axis=1))
    plt.plot(np.sum(rolling_pred_array,axis=1))
    plt.legend(['Actual','Predicted'],fontsize=15)
    plt.ylabel('Total Demand',fontsize=20)
    plt.xlabel('Time (Hour)',fontsize=20)
    plt.savefig('Figures/'+model_name+'_'+dataset+'_'+'ACTUAL VS PREDICTED DEMAND ROLLING')
    plt.close()

    print('Rolling MAE:',mae(rolling_true,rolling_pred_array))
    print('Rolling RMSE:',math.sqrt(mse(rolling_true,rolling_pred_array)))
    print('Rolling MAPE:',MAPE(rolling_true,rolling_pred_array)) #change
    print('Rolling MAPE2:',MAPE2(rolling_true,rolling_pred_array))
    print("-"*100)

if __name__ == "__main__":
    main()