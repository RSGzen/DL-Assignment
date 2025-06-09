import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os

# =======================
# CUSTOM EFFICIENTNET IMPLEMENTATION
# =======================

class CustomMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(CustomMBConvBlock, self).__init__()
        self.stride = stride # store the stride for foward pass
        self.expand_ratio = expand_ratio # store the expansion ratio for conditional logic
        
        mid_channels = int(in_channels * expand_ratio)
        
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False) if expand_ratio != 1 else None # add 1x1 convolutional layer to increase the number of channel
        self.bn0 = nn.BatchNorm2d(mid_channels) if expand_ratio != 1 else None # batch normalization for the expand channel
        
        self.depthwise_conv = nn.Conv2d(
            mid_channels if expand_ratio != 1 else in_channels,
            mid_channels if expand_ratio != 1 else in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2, #ensure the output spatial dimension are preserved
            groups=mid_channels if expand_ratio != 1 else in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels if expand_ratio != 1 else in_channels) # batch normalize after the depthwise convolution
        
        se_channels = max(1, int(in_channels * se_ratio))  # Squeeze-and-Excitation (SE) block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(mid_channels if expand_ratio != 1 else in_channels, se_channels, kernel_size=1),# reduce channel to sechannel 
            nn.SiLU(),
            nn.Conv2d(se_channels, mid_channels if expand_ratio != 1 else in_channels, kernel_size=1), # expand back to the original number of channel 
            nn.Sigmoid()
        )
        
        self.reduce_conv = nn.Conv2d( # reduce the number of channel to outchannel using 1x1 convolution
            mid_channels if expand_ratio != 1 else in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) # batch normalization after the reduction convolution
        
        self.use_residual = in_channels == out_channels and stride == 1 # residual connection is apply if the input and output channel match and the stride is 1
        
    def forward(self, x):
        identity = x # store x as identity for the residual connection
        
        if self.expand_ratio != 1: #apply 1x1 convolution , batch normalization and Silu activation 
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = nn.functional.silu(x)
        
        x = self.depthwise_conv(x) # apply depthwise convolution , batch normalization andSilu activation 
        x = self.bn1(x)
        x = nn.functional.silu(x)
        
        se = self.se(x) # compute attention weight through SE block and scales the channel
        x = x * se
        
        x = self.reduce_conv(x) # apply 1x1 convolution and batch normalization to reduce channel
        x = self.bn2(x)
        
        if self.use_residual:
            x = x + identity
        
        return x

class CustomEfficientNet(nn.Module):
    def __init__(self, config, pretrained=False):
        super(CustomEfficientNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, config['stem_channels'], kernel_size=3, stride=2, padding=1, bias=False), # 3x3 convolution with stride 2 (downsampling spatial dimension by 2)
            nn.BatchNorm2d(config['stem_channels']),# normalize the output
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList()
        in_channels = config['stem_channels']
        
        for stage in config['stages']: # iterate over stage in config. where each stage define a group of blocks
            num_layers = stage['num_layers']
            out_channels = stage['out_channels']
            expand_ratio = stage['expand_ratio']
            kernel_size = stage['kernel_size']
            stride = stage['stride']
            
            for i in range(num_layers):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    CustomMBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        se_ratio=0.25
                    )
                )
                in_channels = out_channels
        
        self.head = nn.Sequential( #final convolutional layer to prepare features for pooling 
            nn.Conv2d(in_channels, config['head_channels'], kernel_size=1, bias=False),
            nn.BatchNorm2d(config['head_channels']),
            nn.SiLU()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1) # average pooling to reduce spatial dimension to 1x1
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config['head_channels'], 1000)
        )
        
        if not pretrained:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): #set weight to 1 and bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x) 
        for block in self.blocks:
            x = block(x) 
        x = self.head(x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x) 
        return x

# =======================
# MODIFIED EFFICIENTNETFER CLASS
# =======================

class EfficientNetFER(nn.Module):
    def __init__(self, num_classes=7, pretrained=False, device='cpu', custom_fc_dims=[512, 256], 
                 dropout=0.3, use_post_conv=False, efficientnet_config=None):
        super(EfficientNetFER, self).__init__()
        self.device = device
        self.use_post_conv = use_post_conv
        
        default_config = {
            'stem_channels': 32, # number of output channel for initial convolutional layer
            'head_channels': 1280, # number of channel before the final classifier
            'stages': [
                {'num_layers': 1, 'out_channels': 16, 'expand_ratio': 1, 'kernel_size': 3, 'stride': 1},
                {'num_layers': 2, 'out_channels': 24, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2},
                {'num_layers': 2, 'out_channels': 40, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 2},
                {'num_layers': 3, 'out_channels': 80, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2},
                {'num_layers': 3, 'out_channels': 112, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 1},
                {'num_layers': 4, 'out_channels': 192, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 2},
                {'num_layers': 1, 'out_channels': 320, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1},
            ]
        }
        
        self.config = efficientnet_config if efficientnet_config else default_config
        
        self.efficientnet = CustomEfficientNet(self.config, pretrained=pretrained)
        self.efficientnet.to(device)
        
        for param in self.efficientnet.parameters():
            param.requires_grad = False # disable gradient computation for all Efficientnet parameter
        
        if self.use_post_conv:
            self.post_conv = nn.Sequential(# additional convolutional block is added after the efficientnet head
                nn.Conv2d(self.config['head_channels'], 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512), 
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1) # reduce the spatial dimension to 1x1 (global average pooling)
            ).to(device)
            fc_input_features = 512
        else:
            fc_input_features = self.config['head_channels'] # 1280
        
        layers = [nn.Dropout(dropout)] # dropout to prevent overfitting
        in_dim = fc_input_features
        for out_dim in custom_fc_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim 
        layers.append(nn.Linear(in_dim, num_classes)) # set a final linear to map the num class
        
        self.classifier = nn.Sequential(*layers).to(device)
    
    def forward(self, x):
        x = self.efficientnet.stem(x)
        for block in self.efficientnet.blocks:
            x = block(x)# iterate through each block , applying the efficientnet architecture
        x = self.efficientnet.head(x)
        
        if self.use_post_conv:
            x = self.post_conv(x)
        else:
            x = self.efficientnet.avgpool(x)
        
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x
    
    def unfreeze_layers(self, num_layers=5):# unfreeze specific layer for finet tuning , allowmthe parameter to be updated during training
        for param in self.classifier.parameters():
            param.requires_grad = True
        layers = list(self.efficientnet.blocks[-num_layers:]) + [self.efficientnet.head] # select  the last numlayers blocks of the efficient and head
        if self.use_post_conv:
            layers.append(self.post_conv)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True 

# =======================
# ORIGINAL TRAINING
# =======================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda', patience=5):
    #criterion (loss function)
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train() # put layer to training mode ( dropout/batch-norm)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) # compute the loss between prediction and true label 
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset) 
        epoch_acc = running_corrects.double() / len(train_loader.dataset) 
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():# disable the gradient computation as gradient are not needed during validation
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) # get predicted class indices
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_loss / len(val_loader.dataset) 
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset) 
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_epoch_loss) # use validation loss to decide whether to reduce learning rate 
        else:
            scheduler.step() # update the learning rate based on the epoch count 
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch - best_epoch > patience:
            print(f'Early stopping at epoch {epoch}') # if the validation accuracy hasnt improved , training stop to prevent overfitting 
            break
    
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    return history

# =======================
# MODIFIED TRANSFORMS FOR GRAYSCALE TO RGB
# =======================

def get_transforms(train_dir):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [0, 1] float tensor
    ])

    dataset = ImageFolder(train_dir, transform=transform)
    # imagefolder read image from directory and transfrom Totensor
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # datafolder to iterate over dataset in batches , process 32 image at a time . This balance memory usage and computation speed
    
    mean = 0.0
    std = 0.0
    total_samples = 0


    for batch in dataloader:
        images = batch[0]  # Extract images
        batch_size = images.size(0)  # get number of image in current batch
        mean += images.mean() * batch_size
        std += images.std() * batch_size
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples

    print(f"Computed Mean: {mean:.4f}")
    print(f"Computed Std: {std:.4f}")

    
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),               # Resize to match EfficientNet-B0 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])  # Normalize across 3 channels
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),               
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])  
    ])
    return train_transform, val_transform
# =======================
# MODIFIED MAIN FUNCTION
# =======================

def main():
    batch_size = 128 
    num_epochs = 50 # number of time entire training dataset is passed through model
    learning_rate = 0.001
    num_classes = 7
    
    if torch.cuda.is_available():# determine to use GPU or CPU for training 
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    base_dir = os.getcwd()
    train_dir = os.path.join(base_dir, "processed_data", "train")
    test_dir = os.path.join(base_dir, "processed_data", "test")

    train_transform, val_transform = get_transforms(train_dir)
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(test_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    custom_config = {
        'stem_channels': 32, # number of output channel in the initial layer of the network. Stem is the first convolutional layer that process the input image
        'head_channels': 1280, #number of channels in the final feature map 
        'stages': [
            {'num_layers': 1, 'out_channels': 16, 'expand_ratio': 1, 'kernel_size': 3, 'stride': 1},
            {'num_layers': 2, 'out_channels': 24, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2},
            {'num_layers': 2, 'out_channels': 40, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 2},
            {'num_layers': 3, 'out_channels': 80, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2},
            {'num_layers': 3, 'out_channels': 112, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 1},
            {'num_layers': 4, 'out_channels': 192, 'expand_ratio': 6, 'kernel_size': 5, 'stride': 2},
            {'num_layers': 1, 'out_channels': 320, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1},
        ]
    }
    
    model = EfficientNetFER(
        num_classes=num_classes,
        device=device,
        pretrained=False,
        custom_fc_dims=[512, 256],
        dropout=0.4,
        use_post_conv=True,
        efficientnet_config=custom_config
    )
    print(model.efficientnet.blocks)
    criterion = nn.CrossEntropyLoss()# combine log softmax and negative log likelihood loss , suitable for multi class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)#reduce learning rate by a factor of 0.1 if validation loss doesnt improve for 3 epochs. Help model converge better
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                          num_epochs=num_epochs, device=device)

    
    print("Starting fine-tuning...")
    model.unfreeze_layers(num_layers=5)#unfreeze the last 5 layers to allow them to adapt the FER task
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate / 10)# create a new optimizer only for unfrozen parameter with reduced learning rate to make more smaller and more precise update
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    history_ft = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                             num_epochs=num_epochs // 2, device=device)
    
    for key in history:
        history[key].extend(history_ft[key])# combine the training and fine tuning histories into single history dictionary .
    
    print("Training complete.")

if __name__ == '__main__':
    main()
