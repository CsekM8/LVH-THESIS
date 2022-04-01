import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from AutoEncoder.ae_dataset import AEDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import time
import copy
from ae_model import ConvAE


# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def AutoEncoderTrain(model, loaders, NUM_EPOCHS):
    train_loss = []
    val_loss = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    randomErase = transforms.RandomErasing(value=1, inplace=True)
    for epoch in range(NUM_EPOCHS):
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for data in loaders[phase]:
                originalNoisedTensor = data + torch.randn_like(data) * 0.09
                originalNoisedTensor = torch.clamp(originalNoisedTensor, min=0, max=1)
                erasedNoisedTensor = originalNoisedTensor.clone()
                for i in range(erasedNoisedTensor.shape[0]):
                    randomErase(erasedNoisedTensor[i])
                originalNoisedTensor = originalNoisedTensor.to(device)
                erasedNoisedTensor = erasedNoisedTensor.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(erasedNoisedTensor)
                    # if phase == 'val':
                    #     print('Output shape: {}'.format(outputs.shape))
                    loss = criterion(outputs, originalNoisedTensor)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * data.size(0)

            loss = running_loss / len(loaders[phase].dataset)
            print('Epoch {} of {}, {} Loss: {:.4f}'.format(
                epoch + 1, NUM_EPOCHS, phase, loss))

            # if phase == 'val' and loss < best_loss:
            #     best_loss = loss
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_loss.append(loss)
            if phase == 'train' and loss < best_loss:
                train_loss.append(loss)
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
            # if epoch % 5 == 0 and phase == 'val':
            #     save_image(outputs.view(outputs.size(0), 1, 168, 168).cpu().data,
            #                './AE_IMAGES/recon_conv_ae_image{}.png'.format(epoch))
            #     save_image(erasedNoisedTensor.view(outputs.size(0), 1, 168, 168).cpu().data,
            #                './AE_IMAGES/original_conv_ae_image{}.png'.format(epoch))
            if epoch % 5 == 0 and phase == 'train':
                save_image(outputs.view(outputs.size(0), 1, 128, 128).cpu().data,
                           './AE_IMAGES/recon_conv_ae_image{}.png'.format(epoch))
                save_image(erasedNoisedTensor.view(outputs.size(0), 1, 128, 128).cpu().data,
                           './AE_IMAGES/original_conv_ae_image{}.png'.format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:5f}'.format(best_loss))
    model.load_state_dict(best_model_wts)

    return model, train_loss, val_loss

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

def test_image_reconstruction(model, testloader):
    if not os.path.exists('../recon'):
        os.makedirs('recon')

    i = 0

    model.eval()

    with torch.no_grad():
        for data in testloader:
            img = data
            original = img.view(img.size(0), 1, 224, 224)
            save_image(original, './recon/original{}.png'.format(i))
            img = img.to(device)
            outputs = model(img)
            outputs = outputs.view(outputs.size(0), 1, 224, 224).cpu().data
            save_image(outputs, './recon/reconstruction{}.png'.format(i))
            difference = torch.abs(original - outputs)
            save_image(difference, './recon/difference{}.png'.format(i))
            i += 1

if __name__ == "__main__":
    # constants
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.005
    BATCH_SIZE = 100

    print("Loading loaders")

    trainset = AEDataset('D:\BME/7felev/Szakdolgozat/whole_dataset/CH2AutoEncoder/training')
    # testset = AEDataset('D:\BME/7felev/Szakdolgozat/whole_dataset/SAAutoEncoder/test')
    # contrastset = AEDataset('D:\BME/7felev/Szakdolgozat/whole_dataset/CH2AutoEncoder/contrast')

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    # testloader = DataLoader(
    #     testset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True
    # )
    testloader = None
    # contrastloader = DataLoader(
    #     contrastset,
    #     batch_size=1,
    #     shuffle=True
    # )

    loaders = {'train': trainloader, 'val': testloader}

    print("Finished loaders")

    AEmodel = ConvAE('B')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(AEmodel.parameters(), lr=LEARNING_RATE, amsgrad=True)

    # get the computation device
    device = get_device()

    AEmodel.to(device)

    if not os.path.exists('./AE_IMAGES'):
        os.makedirs('./AE_IMAGES')

    # if not os.path.exists('./transform_test'):
    #     os.makedirs('./transform_test')

    print("Starting training")

    # train the network
    AEmodel, final_train_loss, final_val_loss = AutoEncoderTrain(AEmodel, loaders, NUM_EPOCHS)

    if not os.path.exists('./graph'):
        os.makedirs('./graph')

    plt.figure()
    plt.plot(final_train_loss, label="Training")
    # plt.plot(final_val_loss, label="Validation")
    plt.title('Loss vs Num. of Training Epochs')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./graph/loss.png')
    plt.close()

    # AEmodel.load_state_dict(torch.load("./AEweights/conv_aeweights"))

    # test the network
    # test_image_reconstruction(AEmodel, contrastloader)

    # save weights
    if not os.path.exists('./AEweights'):
        os.makedirs('./AEweights')
    torch.save(AEmodel.state_dict(), "./AEweights/conv_aeweights")
