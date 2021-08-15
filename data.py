from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy.random import randn, randint
from numpy import ones, zeros
from numpy import asarray, savez_compressed, load
from os import listdir, path
from PIL import Image
from tensorflow.keras.models import Sequential

def loadDataset(doPrepareDataset=False, sampleNum=50000, directory='img_align_celeba/', file='img_align_celeba.npz'):
    datasetFilePath = path.join(directory, file)
    if doPrepareDataset:
        prepareDataset(path.join(directory, directory), datasetFilePath, sampleNum)

    data = load(datasetFilePath)
    X = data['arr_0'].astype('float32')
    X = (X - 127.5) / 127.5
    return X

def generateRealTrainingSamples(dataset, sampleNum):
    ix = randint(0, dataset.shape[0], sampleNum)
    X = dataset[ix]
    y = ones((sampleNum, 1))
    return X, y

def generateFakeTrainingSamples(generator: Sequential, latentDim, sampleNum):
    xInput = generateLatentPoints(latentDim, sampleNum)
    X = generator.predict(xInput)
    y = zeros((sampleNum, 1))
    return X, y

def generateFakeTrainingGanSamples(latentDim, sampleNum):
    X = generateLatentPoints(latentDim, sampleNum)
    y = ones((sampleNum, 1))
    return X, y

def generateLatentPoints(latentDim, sampleNum):
    xInput = randn(latentDim * sampleNum)
    xInput = xInput.reshape((sampleNum, latentDim))
    return xInput

def extractFace(model, pixels, outputSize=(80,80)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None

    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    facePixels = pixels[y1:y2, x1:x2]
    faceImage = Image.fromarray(facePixels)
    faceImageSized = faceImage.resize(outputSize)
    faceArray = asarray(faceImageSized)
    return faceArray

def plotDataset(dataset, sampleNum, sampleOutputPath='img_align_celeba/sample.png'):
    for i in range(sampleNum * sampleNum):
        pyplot.subplot(sampleNum, sampleNum, i + 1)
        pyplot.axis('off')
        pyplot.imshow(dataset[i])
    
    pyplot.savefig(sampleOutputPath)
    pyplot.close()

def prepareDataset(imageDirectory, outputFilePath, sampleNum):
    model = MTCNN()
    faces = list()
    for filename in listdir(imageDirectory):
        image = Image.open(path.join(imageDirectory, filename))
        imageRgb = image.convert('RGB')
        pixels = asarray(imageRgb)
        face = extractFace(model, pixels)
        if face is None:
            continue

        print(face.shape, len(faces), 'of', sampleNum)
        faces.append(face)
        if len(faces) >= sampleNum:
            break
    
    facesArray = asarray(faces)
    print('Loaded:', facesArray.shape)
    plotDataset(facesArray, 5)
    savez_compressed(outputFilePath, facesArray)
