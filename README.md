# Enhancing Traffic Flow Prediction with CMT Fusion: Convolutional Neural Networks and Transformers

Our model leverages the fusion of Convolutional Neural Networks (CNNs) and Transformers to elevate the accuracy of traffic flow prediction. Designed as a regression model using PyTorch, our innovative approach foretells the upcoming hour of traffic, leveraging insights from the preceding four hours. Utilizing a streamlined data stream approach, our model comprehensively processes the four-hour data sequence in a single iteration, enabling it to seamlessly generate predictions for the fifth hour. Dive into the future of traffic prediction with the synergy of CNNs and Transformers in our innovative CMT model.

## Quick Start Guide

Follow these instructions to swiftly set up and run the project on your local machine for both development and testing purposes. Get started with ease and efficiency.

### Prerequisites

Since its a pytorch model which extensively makes use of the sklearn library make sure to have the following installed     
```pip install pytorch```      
```pip install sklearn```

### Installing

Open a folder in VS Code or any other IDE and simply run the following command in the terminal     
```git clone https://github.com/Mustafa-Ashfaq81/Traffic_Prediction_CMT_NYCTaxi.git```       
The entire model will be cloned on your device and ready to run.

## Running the model

Now to run the model, simply navigate into the folder using           
```cd Traffic_Prediction_CMT_NYCTaxi```     
and finally     
```python3 runthis.py```        
You should be able to see the sizes of datasets at the start and the loss after every epochs. Once done the losses will be printed and a graph will be generated showing the actual and predicted values. Navigate into the Figures folder to see this graph.

### Playing with Parameters

The parameters we have used can be found in Param_Our.py. You can play around and tweak them if you like.

## Built With
### Python Libraries
- os: Operating system interaction      
- math: Mathematical functions   
### Numerical Computing
- numpy: Numerical operations in Python   
- matplotlib.pyplot: Data visualization library  
### Deep Learning Framework
- torch: PyTorch for deep learning    
- torch.optim: Optimization algorithms in PyTorch 
### Machine Learning Utilities
- train_test_split: Splitting datasets for training and testing
- mean_squared_error: Scikit-learn's function for calculating mean squared error
- mean_absolute_error: Scikit-learn's function for calculating mean absolute error
### Custom Modules
- Param_Our: Custom module for parameter configurations
- CMT: Custom module for our CMT model

## Authors

- Mustafa Ashfaq 
- Wajiha Naveed

## Acknowledgments
### CMT: Convolutional Neural Networks Meet Vision Transformers
We express gratitude to the authors of "CMT: Convolutional Neural Networks Meet Vision Transformers" for their groundbreaking hybrid network, addressing performance and computational cost gaps between transformers and convolutional neural networks. Their work, particularly the development of CMTs, inspires our use of a CMT model for advanced traffic prediction.

### DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction
Our gratitude to the authors of "DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction" for their impactful insights on grid-based models and on the utilization of deep learning on spatiotemporal data, has ignited significant interest in urban traffic prediction within the AI and Intelligent Transportation System community.

### ______
We extend our sincere appreciation to _____ for providing the code on CMT, which greatly aided our understanding of the model and how to fuse a transformer and a CNN. His model served as inspiration for our development of a CMT model designed for regression, predicting traffic flow.

