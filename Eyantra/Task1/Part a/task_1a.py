'''
*******************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*******************************
'''

# Team ID:			GG_3895
# Author List:		Ashwin Agrawal, Aditya Waghmare, Soham Pawar, Siddhant Godbole
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
##############################################################

###################### Additional Imports ####################
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
'''
##############################################################

def data_preprocessing(task_1a_dataframe):

	'''
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that
	there are features in the csv file whose values are textual (eg: Industry,
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function
	should return encoded dataframe in which all the textual features are
	numerically labeled.

	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset

	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################

	##########################################################
	label_encoders = {}
	categorical_cols = ['Education', 'City', 'Gender','EverBenched']
	for col in categorical_cols:
			label_encoders[col] = LabelEncoder()
			task_1a_dataframe[col] = label_encoders[col].fit_transform(task_1a_dataframe[col])
	encoded_dataframe = task_1a_dataframe
	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
  '''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to
						numbers starting from zero

	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the
							selected features and second item is the target label
Education	JoiningYear	City	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain	LeaveOrNot
	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''
  x = encoded_dataframe[['Education','JoiningYear','City','PaymentTier','Age','Gender','EverBenched','ExperienceInCurrentDomain']]
  y = encoded_dataframe['LeaveOrNot']
#   scaler = StandardScaler()
#   x = scaler.fit_transform(x)
  features_and_targets = []
  features_and_targets.append(x)
  features_and_targets.append(y)
  return features_and_targets

def load_as_tensors(features_and_targets):

  '''
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training
	and validation, and then load them as as tensors.
	Training of the model requires iterating over the training tensors.
	Hence the training sensors need to be converted to iterable dataset
	object.

	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the
							selected features and second item is the target label

	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in
												 batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''

  X = features_and_targets[0]
  y = features_and_targets[1]

  # # Normalize numerical variables
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Convert data to PyTorch tensors
  X_train = torch.FloatTensor(X_train)
  y_train = torch.FloatTensor(y_train)
  X_test = torch.FloatTensor(X_test)
  y_test = torch.FloatTensor(np.array(y_test))
  train_dataset = TensorDataset(X_train, y_train)
  batch_size = 32
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  tensors_and_iterable_training_data = [X_train, X_test, y_train, y_test, train_loader]

  return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
  '''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers.
	It defines the sequence of operations that transform the input data into
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and
	the output is the prediction of the model based on the input data.

	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
  def _init_(self):
        super(Salary_Predictor, self)._init_()
        self.fc1 = nn.Linear(8, 64)  # First hidden layer with 64 units
        self.dropout1 = nn.Dropout(0.3)  # Dropout for regularization
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer with 32 units
        self.dropout2 = nn.Dropout(0.2)  # Dropout for regularization
        self.output = nn.Linear(32, 1)  # Output layer with 1 unit
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        
  def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)  # ReLU activation for the first hidden layer
        x = self.dropout1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)  # ReLU activation for the second hidden layer
        x = self.dropout2(x)
        x = self.output(x)
        x = self.sigmoid(x)  # Sigmoid activation for the output layer
        return x


def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures
	how well the predictions of a model match the actual target values
	in training data.

	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	loss_function = nn.BCELoss()

	return loss_function

def model_optimizer(model):
  '''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible
	for updating the parameters (weights and biases) in a way that
	minimizes the loss function.

	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  return optimizer

def model_number_of_epochs():
  '''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
  number_of_epochs = 30
  return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
  '''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors
											 and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''
  # Set up learning rate scheduler
  scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce learning rate by half every 20 epochs

  # Training the model
  num_epochs = number_of_epochs

  train_loader = tensors_and_iterable_training_data[4]

  for epoch in range(num_epochs):
      for batch_X, batch_y in train_loader:
          outputs = model(batch_X)
          loss = loss_function(outputs.squeeze(), batch_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      scheduler.step()  # Update learning rate at the end of each epoch

      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

  trained_model = model
  return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
  '''
	Purpose:
	---
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors
											 and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''
  # Evaluate the model
  with torch.no_grad():
      trained_model.eval()
      test_outputs = trained_model(tensors_and_iterable_training_data[1])
      predicted = (test_outputs.squeeze() >= 0.5).float()
      accuracy = (predicted == tensors_and_iterable_training_data[3]).sum().item() / tensors_and_iterable_training_data[3].size(0)
      # print(f'Test Accuracy: {accuracy * 100:.2f}%')
      model_accuracy = accuracy * 100
      return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and
	# converting it to a pandas Dataframe
	task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy:.2f}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
