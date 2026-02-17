from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    def __init__(
        self,
        input_shape,
        categories_count,
        conv_filters=(16, 32, 32),
        dense_units=(64,),
        dropout_rates=(),
        learning_rate=0.001,
    ):
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        super().__init__(input_shape, categories_count)

    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        model_layers = [Rescaling(1. / 255, input_shape=input_shape)]

        for filter_count in self.conv_filters:
            model_layers.append(
                layers.Conv2D(filter_count, (3, 3), activation='relu', padding='same')
            )
            model_layers.append(layers.MaxPooling2D(pool_size=(2, 2)))

        model_layers.append(layers.Flatten())

        for index, unit_count in enumerate(self.dense_units):
            model_layers.append(layers.Dense(unit_count, activation='relu'))
            if index < len(self.dropout_rates):
                model_layers.append(layers.Dropout(self.dropout_rates[index]))

        model_layers.append(layers.Dense(categories_count, activation='softmax'))

        self.model = Sequential(model_layers)
            
            
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )