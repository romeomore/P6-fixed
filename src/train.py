import numpy as np
import os
from preprocess import get_datasets
from models.basic_model import BasicModel
from config import image_size
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    #
    # Your code should change the number of epochs
    epochs = 30
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()

    search_space = [
        {
            'conv_filters': (16, 32, 64),
            'dense_units': (64,),
            'dropout_rates': (),
            'learning_rate': 0.001,
        },
        {
            'conv_filters': (16, 32, 64),
            'dense_units': (128,),
            'dropout_rates': (0.3,),
            'learning_rate': 0.001,
        },
        {
            'conv_filters': (16, 32, 64),
            'dense_units': (128, 64),
            'dropout_rates': (0.3, 0.3),
            'learning_rate': 0.0005,
        },
        {
            'conv_filters': (32, 64, 64),
            'dense_units': (128,),
            'dropout_rates': (0.4,),
            'learning_rate': 0.0005,
        },
        {
            'conv_filters': (16, 32, 64, 64),
            'dense_units': (128,),
            'dropout_rates': (0.4,),
            'learning_rate': 0.0005,
        },
        {
            'conv_filters': (16, 32, 64, 64),
            'dense_units': (256, 64),
            'dropout_rates': (0.4, 0.3),
            'learning_rate': 0.0003,
        },
    ]

    best_model = None
    best_history = None
    best_result = None

    for config_index, config in enumerate(search_space, start=1):
        print('\n* Training config {}/{}'.format(config_index, len(search_space)))
        print('  - conv_filters: {}'.format(config['conv_filters']))
        print('  - dense_units: {}'.format(config['dense_units']))
        print('  - dropout_rates: {}'.format(config['dropout_rates']))
        print('  - learning_rate: {}'.format(config['learning_rate']))

        model = BasicModel(
            input_shape,
            categories_count,
            conv_filters=config['conv_filters'],
            dense_units=config['dense_units'],
            dropout_rates=config['dropout_rates'],
            learning_rate=config['learning_rate'],
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
        )

        history = model.train_model(
            train_dataset,
            validation_dataset,
            epochs,
            callbacks=[early_stopping],
        )

        best_epoch = int(np.argmax(history.history['val_accuracy']) + 1)
        best_val_accuracy = float(np.max(history.history['val_accuracy']))
        test_loss, test_accuracy = model.model.evaluate(test_dataset, verbose='auto')

        result = {
            'config': config,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
        }

        print('  - best validation accuracy: {:.4f} at epoch {}'.format(best_val_accuracy, best_epoch))
        print('  - test accuracy: {:.4f}'.format(test_accuracy))

        if best_result is None or result['best_val_accuracy'] > best_result['best_val_accuracy']:
            best_result = result
            best_model = model
            best_history = history

    print('\n* Best configuration selected')
    print(best_result)
    print('* Evaluating best_model')
    best_model.evaluate(test_dataset)
    print('* Confusion Matrix for best_model')
    print(best_model.get_confusion_matrix(test_dataset))

    os.makedirs('results', exist_ok=True)
    model_name = 'step6_best_model_{}_epochs_timestamp_{}'.format(epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    best_model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), best_history)
    np.save('results/{}_meta.npy'.format(model_name), best_result)

    print('* Best model saved as {}'.format(filename))
    plot_history(best_history)
