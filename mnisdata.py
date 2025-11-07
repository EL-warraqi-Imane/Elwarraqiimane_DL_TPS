import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

class MNISTData:
    def __init__(self):
        # Transformation : tensor + normalisation
        transform = transforms.Compose([
            transforms.ToTensor(),  # 28x28 -> tensor
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Chargement du dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Images en vecteurs (28*28=784)
        self._Xtrain = train_dataset.data.reshape(-1, 28*28).float() / 255.0
        self._Xtest  = test_dataset.data.reshape(-1, 28*28).float() / 255.0
        
        # Labels en one-hot
        self._Ytrain = F.one_hot(train_dataset.targets, num_classes=10).float()
        self._Ytest  = F.one_hot(test_dataset.targets, num_classes=10).float()
        
        # Accumulateurs pour loss / accuracy
        self.loss_train = []
        self.loss_test  = []
        self.acc_train  = []
        self.acc_test   = []

    # Getter pour compatibilité avec code existant
    def __getattr__(self, key):
        if key == "Xtrain": return self._Xtrain
        if key == "Xtest":  return self._Xtest
        if key == "Ytrain": return self._Ytrain
        if key == "Ytest":  return self._Ytest
        return None

    # Option : afficher les courbes
    def plot_loss(self, loss_train, loss_test, acc_train, acc_test):
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.acc_train, label="Acc Train")
        plt.plot(self.acc_test, label="Acc Test")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.loss_train, label="Loss Train")
        plt.plot(self.loss_test, label="Loss Test")
        plt.legend()
        plt.show()
