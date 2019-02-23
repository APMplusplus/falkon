import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

bsz = 128
max_epochs = 100
beta = 4

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=bsz, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=bsz, shuffle=True)


class VAE(nn.Module):

   def __init__(self):
       super(VAE, self).__init__()

       self.encoder_fc = nn.Linear(784, 400)
       self.mean_fc = nn.Linear(400, 20)
       self.logvar_fc = nn.Linear(400,20)
       self.prefinal_fc = nn.Linear(20, 400)
       self.final_fc = nn.Linear(400, 784)

   def encoder(self, x):
       encoded = torch.relu(self.encoder_fc(x))
       mu = self.mean_fc(encoded)
       log_var = self.logvar_fc(encoded)

       return mu, log_var

   def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

   def decoder(self, z):
        decoded = F.relu(self.prefinal_fc(z))
        return torch.sigmoid(self.final_fc(decoded))

   def forward(self, x):
        x = x.view(-1, 784)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        #print("Shape of z from reparameterization: ", z.shape)
        return self.decoder(z), mu, log_var


model = VAE()
if torch.cuda.is_available():
  model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

def val(epoch):

   model.eval()
   z = torch.rand(1,20).cuda()
   with torch.no_grad():
      a = model.decoder(z).view(1,1,28,28)
      save_image(a.cpu(),'/tmp/reconstruction_' + str(epoch) + '.png')
      

def train(epoch):

   model.train()
   train_loss = 0
 
   for i, (data, _) in enumerate(train_loader):

       if torch.cuda.is_available():
          data = data.cuda()
       optimizer.zero_grad()
       recon_batch, mu, logvar = model(data)
       loss = loss_function(recon_batch, data, mu, logvar)
       loss.backward()
       train_loss += loss.item()
       torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
       optimizer.step()

       #if i % 10 == 1:
       #   print(train_loss/(i+1))

   return train_loss/(len(train_loader.dataset))



def main():

  for epoch in range(max_epochs):
      train_loss =  train(epoch)
      val(epoch)
      print("Train loss after ", epoch, " epochs: ", train_loss)


main()
