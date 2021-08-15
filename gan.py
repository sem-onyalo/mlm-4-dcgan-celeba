from generator import createGenerator
from discriminator import createDiscriminator
from data import generateRealTrainingSamples, generateFakeTrainingSamples, generateFakeTrainingGanSamples
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def createGan(discriminator: Sequential, generator: Sequential):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def train(discriminator: Sequential, generator: Sequential, gan: Sequential, dataset, latentDim, epochNum=100, batchNum=128, evalFreq=10):
    batchesPerEpoch = int(dataset.shape[0] / batchNum)
    halfBatch = int(batchesPerEpoch / 2)
    for i in range(epochNum):
        for j in range(batchesPerEpoch):
            xReal, yReal = generateRealTrainingSamples(dataset, halfBatch)
            dLossReal, _ = discriminator.train_on_batch(xReal, yReal)

            xFake, yFake = generateFakeTrainingSamples(generator, latentDim, halfBatch)
            dLossFake, _ = discriminator.train_on_batch(xFake, yFake)

            xGan, yGan = generateFakeTrainingGanSamples(latentDim, batchNum)
            gLoss = gan.train_on_batch(xGan, yGan)

            print('>%d, %d/%d, dr=%.3f, df=%.3f, g=%.3f' % 
                (i + 1, j + 1, batchesPerEpoch, dLossReal, dLossFake, gLoss))

        if i == 0 or (i + 1) % evalFreq == 0:
            evaluatePerformance(i, discriminator, generator, dataset, latentDim)

def evaluatePerformance(epoch, discriminator: Sequential, generator: Sequential, dataset, latentDim, numSamples=150):
    xReal, yReal = generateRealTrainingSamples(dataset, numSamples)
    _, accReal = discriminator.evaluate(xReal, yReal, verbose=0)

    xFake, yFake = generateFakeTrainingSamples(generator, latentDim, numSamples)
    _, accFake = discriminator.evaluate(xFake, yFake, verbose=0)

    print(epoch, accReal, accFake)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accReal * 100, accFake * 100))

    savePlot(xFake, epoch)

    filename = 'eval/generated_model%03d.h5' % (epoch + 1)
    generator.save(filename)

def savePlot(examples, epoch, n=7):
    # scale from -1,1 to 0,1
    scaledExamples = (examples + 1) / 2.0

    for i in range(n * n):
        pyplot.subplot(n, n, i + 1)
        pyplot.axis('off')
        pyplot.imshow(scaledExamples[i])

    filename = 'eval/generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


if __name__ == '__main__':
    generator = createGenerator()
    discriminator = createDiscriminator()
    gan = createGan(discriminator, generator)
    gan.summary()