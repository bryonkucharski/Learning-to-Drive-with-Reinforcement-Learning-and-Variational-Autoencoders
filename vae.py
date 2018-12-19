import os
import torch
import torch.utils.data
from torch import nn, optim
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from hyperdash import Experiment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
class FC_VAE(nn.Module):
    '''
    Thanks to https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
    '''
    def __init__(self, input_size, NUM_Z, HIDDEN_1, batch_size, learning_rate = 1e-3, CUDA = True):
        super(FC_VAE, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        # ENCODER
        self.fc1 = nn.Linear(input_size, HIDDEN_1)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(HIDDEN_1, NUM_Z)  # mu layer
        self.fc22 = nn.Linear(HIDDEN_1, NUM_Z)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(NUM_Z, HIDDEN_1)

        self.fc4 = nn.Linear(HIDDEN_1, input_size)
        self.sigmoid = nn.Sigmoid()

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)
        self.CUDA = CUDA

    def encode(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        z1 = self.fc21(out)
        z2 = self.fc22(out)
        return z1, z2

    def decode(self, z):
        out = self.fc3(z)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out
    
    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z =  eps.mul(std).add_(mu)
            return z
        else:
            return mu

    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

    def predict(self,s):
        self.eval()
        data = torch.from_numpy(s).float().to(device)
        #data = Variable(s, volatile=True)
        mu, logvar = self.encode(data.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z.data.cpu().numpy()


    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        KLD /= self.batch_size * self.input_size

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD

    def train_model(self, epoch, train_loader, LOG_INTERVAL):
        self.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if self.CUDA:
                data = data.cuda()
            self.optim.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = self.forward(data)
            # calculate scalar loss
            loss = self.loss_function(recon_batch, data, mu, logvar)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.data[0]
            self.optim.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        return train_loss

    def test_model(self, epoch, test_loader):
        self.eval()
        test_loss = 0

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if self.CUDA:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.forward(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits

            comparison = torch.cat([data[:n],
                                    recon_batch.view(recon_batch.size()[0], 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                        'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class Conv_VAE(nn.Module):

    def __init__(self, learning_rate = 1e-3,image_channels=3, h_dim=1024, z_dim=32, CUDA = True, batchsize = 128):
        super(Conv_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)
        self.CUDA = CUDA
        self.batch_size = batchsize

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #KLD /= self.batch_size * self.input_size

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD

    def train_model(self, epoch, train_loader, LOG_INTERVAL):
        self.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if self.CUDA:
                data = data.cuda()
            self.optim.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = self.forward(data)
            # calculate scalar loss
            loss = self.loss_function(recon_batch, data, mu, logvar)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.data[0]
            self.optim.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        return train_loss / len(train_loader.dataset)

    def test_model(self, epoch, test_loader):
        self.eval()
        test_loss = 0

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if self.CUDA:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.forward(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits

            comparison = torch.cat([data[:n],
                                    recon_batch.view(recon_batch.size()[0], 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                       'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss




def load_dataset(path,batch_size):
    data_path = path
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def main():
    exp = Experiment("duckietown_vae")
    model_name = 'models/duckietown_vae_fc_model.pt'
    # changed configuration to this instead of argparse for easier interaction
    CUDA = True
    SEED = 1
    BATCH_SIZE = 32
    LOG_INTERVAL = 10
    EPOCHS = 25

    # connections through the autoencoder bottleneck
    # in the pytorch VAE example, this is 20
    ZDIMS = 100



    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)

    # DataLoader instances will load tensors directly into GPU memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    # Download or load downloaded MNIST dataset
    # shuffle data at every epoch
    #train_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('data', train=True, download=True,
                    #transform=transforms.ToTensor()),
        #batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # Same for test data
    #test_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        #batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_loader = load_dataset('images/train', BATCH_SIZE)
    test_loader = load_dataset('images/test', BATCH_SIZE)

    vae = FC_VAE(  input_size = 64*64*3,
                HIDDEN_1 = 400, 
                batch_size = BATCH_SIZE,
                NUM_Z = ZDIMS, 
                learning_rate = 1e-3,
                CUDA=CUDA)

    #vae = Conv_VAE(batchsize=BATCH_SIZE)
    vae.cuda()
    total_test_loss = []
    total_train_loss = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = vae.train_model(epoch, train_loader,LOG_INTERVAL)
        total_train_loss.append(train_loss)
        exp.metric("train_loss", train_loss.item())
        test_loss = vae.test_model(epoch, test_loader)
        total_test_loss.append(test_loss)
        exp.metric("test_loss", test_loss.item())
        torch.save(vae.state_dict(), model_name)

    
    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space

    #sample = Variable(torch.randn(64, ZDIMS))
    #if CUDA:
        #sample = sample.cuda()
    #sample = vae.decode(sample).cpu()

    # save out as an 8x8 matrix of MNIST digits
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    #save_image(sample.data.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    line, = ax.plot(range(EPOCHS),total_train_loss, color='blue', lw=2,label = 'Train')
    line, = ax.plot(range(EPOCHS),total_test_loss, color='red', lw=2, label = 'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.legend()
    plt.show()
    

#main()