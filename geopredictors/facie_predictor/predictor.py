'''
This module contains the FaciesPredictor class, which can be used to 
predict the facies of a given dataset.
'''
import random
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score
from livelossplot import PlotLosses
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader,random_split

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a 
    fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False

    return True

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


class NumpyImageDataset(Dataset):
    '''
    Numpy Dataset class.
    '''
    def __init__(self, folder_path, transform=None):
        '''
        Initializes the NumpyImageDataset class.
        Input:
            folder_path: str
                Path to the folder where the images are stored.
            transform: PyTorch transform
                The transform to apply to the images.
        '''
        self.folder_path = folder_path
        self.transform = transform

        # List subdirectories (categories)
        self.categories = sorted(os.listdir(folder_path))

        # List all .npy files in each subdirectory
        self.file_paths = []
        for category in self.categories:
            category_path = os.path.join(folder_path, category)
            file_names = os.listdir(category_path)
            file_paths = [os.path.join(category_path, file_name) for
                          file_name in file_names if file_name.endswith('.npy')]
            self.file_paths.extend(file_paths)

    def __len__(self):
        '''
        Returns the length of the dataset.
        Output:
            len: int
                The length of the dataset.
        '''
        return len(self.file_paths)

    def __getitem__(self, idx):
        '''
        Returns a tuple (image, label) where image is a PyTorch tensor
        and label is a PyTorch tensor.
        Input:
            idx: int
                The index of the sample.
        Output:
            image: PyTorch tensor
                The image tensor.
            label: PyTorch tensor
                The label tensor.
        '''
        # Load .npy file
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        # Convert numpy array to PIL Image
        image = Image.fromarray(data)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Get label from the subdirectory name
        label = self.categories.index(os.path.basename(os.path.dirname(file_path)))

        return image, label

class CustomResNet(nn.Module):
    '''
    Our custom ResNet model.
    '''
    def __init__(self, num_classes=6, leak=0.2,dropout=True,dropout_rate=0.15):
        '''
        Initializes the CustomResNet class.
        Input:
            num_classes: int
                The number of classes.
            leak: float
                The negative slope of the LeakyReLU activation function.
            dropout: bool
                Whether to use dropout or not.
            dropout_rate: float
                The dropout rate.
        '''
        super(CustomResNet, self).__init__()
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Load a pre-trained ResNet
        self.resnet = models.resnet34(pretrained=True)

        # Remove the fully connected layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Custom Convolutional layers
        # 512 channels from ResNet's last layer
        self.conv1 = nn.Conv2d(512, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.leakyrelu = nn.LeakyReLU(negative_slope=leak)
        self.flatten = nn.Flatten()
        #add a dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Fully connected layer for classification
        self.fc = nn.Linear(480, num_classes)  # Assuming the output size after pooling is 7x7

    def forward(self, x):
        '''
        Forward pass of the network.
        Input:
            x: PyTorch tensor
                The input tensor.
        Output:
            x: PyTorch tensor
                The output tensor.
        '''
        # Pass input through ResNet layers
        x = self.resnet(x)
        # Pass through custom layers
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        x = self.leakyrelu(self.conv3(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.leakyrelu(self.conv9(x))

        # Flatten and pass through the fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

class FaciesPredictor():
    '''
    The heart of the FaciesPredictor package.
    '''
    def __init__(self,dropout=True,dropout_rate=0.15):
        '''
        Initializes the FaciesPredictor class. This will help predict
        the facies of a given dataset.
        Input:
            dropout: bool
                Whether to use dropout or not.
            dropout_rate: float
                The dropout rate.
        '''
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.model = CustomResNet(dropout=self.dropout,dropout_rate=self.dropout_rate)

    def train(self,train_path,epochs=20,lr=0.001,weight_decay=1e-4,scheduler=False):
        '''
        Trains the model.
        Input:
            train_path: str
                Path to the folder where the images are stored.
        '''
        self.train_loader, self.val_loader = self.load_and_split(train_path)
        self.lr = lr
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.model = self.train_model()
    
    def print_model(self):
        '''
        Prints the model architecture.
        '''
        print(self.model)
        #print also the type of the model
        print(type(self.model))


    def load_model(self,model_path):
        '''
        Loads a model from a file.
        Input:
            model_path: str
                Path to the file containing the model.
        Output:
            model: PyTorch model
                The loaded model.
        '''
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")

    def load_and_split(self,path):
        '''
        Loads the images and splits them into training and validation sets.
        Input:
            path: str
                Path to the folder where the images are stored.
        Output:
            train_loader: PyTorch DataLoader
                The data loader that iterates over the training dataset.
            val_loader: PyTorch DataLoader
                The data loader that iterates over the validation dataset.
            '''
        self.transform = transforms.Compose([
        transforms.Resize((70, 300)),  # Adjust the size as needed
        transforms.ToTensor()])
        dataset = NumpyImageDataset(path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.2 * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        return train_loader,val_loader

    def validate(self, model, criterion, data_loader):
        '''
        Computes the loss and accuracy of the model on the validation dataset.
        Input:
            model: PyTorch model
                The model to evaluate.
            criterion: PyTorch loss function
                The loss function used to compute the loss.
            data_loader: PyTorch DataLoader
                The data loader that iterates over the validation dataset.
        Output:
            validation_loss: float
                The average loss over the validation dataset.
            validation_accuracy: float
                The average accuracy over the validation dataset.
        '''
        model.eval()
        validation_loss, validation_accuracy = 0., 0.
        for X, y in data_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                a2 = model(X.view(-1, 3, 70, 300))
                loss = criterion(a2, y)
                validation_loss += loss.detach().item() * X.size(0)
                y_pred = F.log_softmax(a2, dim=1).max(1)[1]
                validation_accuracy += accuracy_score(y.cpu().numpy(),
                                                      y_pred.cpu().numpy()) * X.size(0)

        return validation_loss / len(data_loader.dataset), validation_accuracy / len(data_loader.dataset)

    def trainer(self, model, optimizer, criterion, data_loader):
        '''
        Trains the model for one epoch.
        Input:
            model: PyTorch model
                The model to train.
            optimizer: PyTorch optimizer
                The optimizer used to update the model's weights.
            criterion: PyTorch loss function
                The loss function used to compute the loss.
            data_loader: PyTorch DataLoader
                The data loader that iterates over the training dataset.
        Output:
            train_loss: float
                The average loss over the training dataset.
            train_accuracy: float
                The average accuracy over the training dataset.
                '''
        model.train()
        train_loss, train_accuracy = 0, 0
        total_samples = len(data_loader.dataset)

        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            a2 = model(X.view(-1, 3, 70, 300))
            loss = criterion(a2, y)
            loss.backward()
            train_loss += loss.detach().item() * X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            train_accuracy += accuracy_score(y.cpu().numpy(),
                                             y_pred.detach().cpu().numpy()) * X.size(0)
            optimizer.step()

            # Print progress
            if (i + 1) % 10 == 0:  # Print every 10 iterations
                print(f"Iteration [{i + 1}/{len(data_loader)}] "
                      f"Loss: {train_loss / (i + 1):.4f} "
                      f"Accuracy: {train_accuracy / total_samples:.4f}")
        return train_loss / total_samples, train_accuracy / total_samples

    def train_model(self, seed=42):
        '''
        Trains the model.
        Input:
            seed: int
                Random seed.
        Output:
            model: PyTorch model
                The trained model.
        '''
        set_seed(seed)
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        liveloss = PlotLosses()
        for _ in range(self.epochs):
            logs = {}
            train_loss, train_accuracy = self.trainer(self.model, optimizer, criterion, self.train_loader)

            logs['log loss'] = train_loss
            logs['accuracy'] = train_accuracy

            validation_loss, validation_accuracy = self.validate(self.model, criterion, self.val_loader)
            logs['val_log loss'] = validation_loss
            logs['val_accuracy'] = validation_accuracy
            if self.scheduler:
                scheduler.step()
            liveloss.update(logs)
            liveloss.draw()
        return self.model

    def save(self,save_path):
        '''
        Saves the model to a file.
        Input:
            save_path: str
                Path to the folder where the model will be saved.
        '''
        self.save_path = save_path
        torch.save(self.model.state_dict(), f"{self.save_path}/model_geoprediction.pth")
        print(f"Model saved to {self.save_path}")

    def segment(self,images, depths, rows_per_segment=140):
        '''
        Segments the images and depths into smaller segments.
        Input:
            images: numpy array
                A numpy array containing the images.
            depths: numpy array
                A numpy array containing the depths.
            rows_per_segment: int
                The number of rows per segment.
        Output:
            segmented_images: list
                A list of numpy arrays containing the segmented images.
            segmented_depths: list
                A list of tuples containing the start and end depths of each segment.
        '''
        segmented_images = []
        segmented_depths = []

        # Calculate the number of segments
        num_segments = len(images) // rows_per_segment

        for i in range(num_segments):
            start_idx = i * rows_per_segment
            end_idx = start_idx + rows_per_segment

            # Segment the images
            segmented_image = images[start_idx:end_idx]
            segmented_images.append(segmented_image)

            # Create the depth tuple
            start_depth = depths[start_idx]
            end_depth = depths[end_idx - 1]  # Use end_idx - 1 to get the last depth in the segment
            segmented_depths.append((start_depth, end_depth))

        return segmented_images, segmented_depths
    def transform_segmented_images(self,segmented_images):
        '''
        Transforms the segmented images to tensors, and makes them
        compatible with the model.
        Input:
            segmented_images: list
                A list of numpy arrays containing the segmented images.
        Output:
            transformed_images: list
                A list of tensors containing the transformed images.
        '''
        transformed_images = []
        self.transform = transforms.Compose([
            transforms.Resize((70, 300)),
            transforms.ToTensor(),
        ])
        for image in segmented_images:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            # Apply the transformation
            transformed_image = self.transform(pil_image)
            transformed_images.append(transformed_image)

        return transformed_images

    def predict(self,npy_image_path,npy_depth_path):
        '''
        Predicts the facies of a given image.
        Input:
            npy_image_path: str
                Path to the folder where the images are stored.
            npy_depth_path: str
                Path to the folder where the depths are stored.
        Output:
            results_df: pandas DataFrame
                A pandas DataFrame containing the start and end 
                depths of each segment, and the predicted facies.
        '''
        images = np.load(npy_image_path)
        depths = np.load(npy_depth_path)
        self.segmented_images, self.segmented_depths = self.segment(images, depths)
        transformed_images = self.transform_segmented_images(self.segmented_images)
        class_mapping = {0: 'ih', 1: 'is', 2: 'nc', 3: 'os', 4: 's', 5: 'sh'}
        # Assuming transformed_images is a list of tensors
        images_batch = torch.stack(transformed_images)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_batch = images_batch.to(device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(images_batch)
            _, predicted_classes = torch.max(outputs, 1)
        predicted_labels = [class_mapping[idx.item()] for idx in predicted_classes]
        self.results_df = pd.DataFrame(self.segmented_depths, columns=['Start Depth', 'End Depth'])
        self.results_df['Predicted Class'] = predicted_labels
        return self.results_df

    def export(self,export_path,name='predictions'):
        '''
        Exports the predictions to a CSV file.
        Input:
            export_path: str
                Path to the folder where the CSV file will be saved.
            name: str
                Name of the CSV file.
        '''
        self.results_df.to_csv(f'{export_path}/{name}.csv', index=False)
        print(f"Predictions exported to {export_path}")
