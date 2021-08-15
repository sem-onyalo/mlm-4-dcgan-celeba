from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, Reshape
from tensorflow.keras.models import Sequential

def createGenerator(n_inputs=100):
    input_dim = 5
    n_filters = 256
    n_filters_hidden = 128
    n_nodes = n_filters * input_dim * input_dim

    model = Sequential()
    model.add(Dense(n_nodes, input_dim=n_inputs))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((input_dim, input_dim, n_filters)))

    model.add(Conv2DTranspose(n_filters_hidden, (4,4), (2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(n_filters_hidden, (4,4), (2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(n_filters_hidden, (4,4), (2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(n_filters_hidden, (4,4), (2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    return model

if __name__ == '__main__':
    generator = createGenerator()
    generator.summary()