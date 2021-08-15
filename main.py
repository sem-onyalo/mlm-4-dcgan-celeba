'''
DCGAN - CelebA Photographs.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

from data import generateLatentPoints, loadDataset
from generator import createGenerator
from discriminator import createDiscriminator
from gan import createGan, train

if __name__ == '__main__':
    latentDim = 100
    dataset = loadDataset()
    discriminator = createDiscriminator()
    generator = createGenerator(latentDim)
    gan = createGan(discriminator, generator)
    train(discriminator, generator, gan, dataset, latentDim)