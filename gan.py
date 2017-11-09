#!/usr/bin/python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time

from descriminator import D
from generator import G

#takes as parameter a neural network and defines its weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

print("""\n\tA simple example of a:
\t+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+
\t|G|e|n|e|r|a|t|i|v|e| |A|d|v|e|r|s|a|r|i|a|l| |N|e|t|w|o|r|k|
\t+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+
\tCode created by Davidson Mizael based on the course:
\t'Computer Vision A-Z' 
\thttps://www.udemy.com/computer-vision-a-z""")

print("\n")
print("# Starting...")

#sets the size of the image (64x64) and the batch
batchSize = 64
imageSize = 64

#creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

print("# Loading data...")
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)


print("# Starting generator and descriminator...")
netG = G()
netG.apply(weights_init)

netD = D()
netD.apply(weights_init)

#training the DCGANs
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

epochs = 25
for epoch in range(epochs):
    print("# Starting epoch [%d/%d]..." % (epoch, epochs))
    for i, data in enumerate(dataloader, 0):
        start = time.time()
        time.clock()  
        
        #updates the weights of the discriminator nn
        netD.zero_grad()
        
        #trains the discriminator with a real image
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        #trains the discriminator with a fake image
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        ouput = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        #backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        #updates the weights of the generator nn
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG  = criterion(output, target)
        
        #backpropagating the error
        errG.backward()
        optimizerG.step()


        if i % 50 == 0:
            #prints the losses and save the real images and the generated images
            print("# Progress: ")
            print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" % (epoch, epochs, i, len(dataloader), errD.data[0], errG.data[0]))
            
            #calculates the remaining time by taking the seconds that this loop took
            #and multiplying by the loops that still need to run
            remaining = ((time.time() - start) * (len(dataloader) - i)) * (epochs - epoch)
            print("# Estimated remaining time: %s" % (time.strftime("%H:%M:%S", time.gmtime(remaining))))
        
        if i % 100 == 0:
            vutils.save_image(real, "%s/real_samples.png" % "./results", normalize = True)
            vutils.save_image(fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results", epoch), normalize = True)

print ("# Finished.")