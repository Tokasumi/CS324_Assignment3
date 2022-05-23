import argparse
import os
import random
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from clean import clean

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('CURRENT DEVICE: ', DEVICE)


class Generator(nn.Module):
    def __init__(self, latent_space=100, output_dim=28 * 28):
        super(Generator, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(latent_space, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator(nn.Module):
    def __init__(self, in_features=28 * 28):
        super(Discriminator, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.backbone(img)


_CHECKPOINTS = {
    0: 'start',
    4000: 'mid',
    120000: 'last',
}


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args):
    criterion = nn.BCELoss()
    d_loss, g_loss = None, None
    for epoch in range(args.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            if d_loss is not None and g_loss is not None:
                d_stat, g_stat = np.exp(d_loss.item()), np.exp(g_loss.item())
                train_d = random.uniform(0.0, d_stat + g_stat) < d_stat
                # train_g = random.uniform(0.0, d_stat + g_stat) < g_stat
                train_g = True
            else:
                train_d, train_g = True, True

            if train_d:
                # Train Discriminator
                # -------------------
                optimizer_D.zero_grad()

                x_real = imgs.reshape(batch_size, -1).to(DEVICE, torch.float32)
                y_real = torch.autograd.Variable(torch.ones(batch_size, 1)).to(DEVICE, torch.float32)

                z = torch.autograd.Variable(torch.randn(batch_size, args.latent_dim)).to(DEVICE, torch.float32)
                x_fake = generator(z)
                y_fake = torch.autograd.Variable(torch.zeros(batch_size, 1)).to(DEVICE, torch.float32)

                x = torch.cat((x_real, x_fake), 0)
                y = torch.cat((y_real, y_fake), 0)

                d_pred = discriminator(x)
                d_loss = criterion(d_pred, y)

                d_loss.backward()
                optimizer_D.step()

            if train_g:
                # Train Generator
                # ---------------
                optimizer_G.zero_grad()

                z = torch.autograd.Variable(torch.randn(batch_size, args.latent_dim)).to(DEVICE, torch.float32)
                y = torch.autograd.Variable(torch.ones(batch_size, 1)).to(DEVICE, torch.float32)

                g_pred = generator(z)
                d_pred = discriminator(g_pred)
                g_loss = criterion(d_pred, y)

                g_loss.backward()
                optimizer_G.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0 and d_loss is not None and g_loss is not None:
                print(f'EPOCH: {epoch}', 'BATCHES_DONE:', batches_done,
                      'D_LOSS:', d_loss.item(), 'G_LOSS:', g_loss.item())
                if g_pred.shape[0] < 25:
                    print('WARNING: Shape mismatch when saving image.')
                    continue
                save_image(g_pred[:25, :].reshape(25, 1, 28, 28).detach().to('cpu'),
                           'images/batches{}.png'.format(batches_done),
                           nrow=5,
                           normalize=True)
                pass

                if batches_done in _CHECKPOINTS:
                    torch.save(generator.state_dict(), f'mnist_generator_{_CHECKPOINTS[batches_done]}.pth')


def main(args):
    # Create o_gate image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr) if args.adam else torch.optim.RMSprop(
        generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr) if args.adam else torch.optim.RMSprop(
        discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), 'mnist_generator_final.pth')


@dataclass
class DefaultArgs:
    n_epoches: int = 200,
    batch_size: int = 64,
    lr: float = 0.0002,
    latent_dim: int = 100,
    save_interval: int = 500,
    adam: bool = False


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save-interval', type=int, default=2000,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam as optimizer, otherwise, use RMSProp.')
    return parser.parse_args()


if __name__ == "__main__":
    clean()
    options = make_args()
    main(options)
