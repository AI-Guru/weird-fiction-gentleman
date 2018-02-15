import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
import os
import urllib
from collections import Counter
import html
import nltk
nltk.download('punkt')
nltk.download('perluniprops')
from nltk import word_tokenize
import pickle
import random
import progressbar
import keras
from keras import models
from keras import layers
from keras import utils

# This tokenizer is nice, but could cause problems.
try:
    from nltk.tokenize.moses import MosesDetokenizer
    detokenizer = MosesDetokenizer()
    use_moses_detokenizer = True
except:
    use_moses_detokenizer = False


# Corpus parameters.
download_anyway = False
corpus_url = "https://archive.org/stream/TheCollectedWorksOfH.p.Lovecraft/The-Collected-Works-of-HP-Lovecraft_djvu.txt"
corpus_path = "lovecraft.txt"

# Preprocessing parameters.
preprocess_anyway = False
preprocessed_corpus_path = "lovecraft_preprocessed.p"
most_common_words_number = 10000

# Training parameters.
train_anyway = False
model_path = "model.h5"
dataset_size = 50000
sequence_length = 30
epochs = 10
batch_size = 128
hidden_size = 1000

# Generation parameters.
generated_sequence_length = 500


def main():
    """ The main-method. Where the fun begins. """

    download_corpus_if_necessary()

    preprocess_corpus_if_necessary()

    train_neural_network()

    generate_texts()


def download_corpus_if_necessary():
    """
    Downloads the corpus either if it is not on the hard-drive or of the
    download is forced.
    """

    if not os.path.exists(corpus_path) or download_anyway == True:
        print("Downloading corpus...")

        # Dowloading content.
        corpus_string = urllib.request.urlopen(corpus_url).read().decode('utf-8')

        # Removing HTML-stuff.
        index = corpus_string.index("<pre>")
        corpus_string = corpus_string[index + 5:]
        index = corpus_string.find("</pre>")
        corpus_string = corpus_string[:index ]
        corpus_string = html.unescape(corpus_string)

        # Write to file.
        corpus_file = open(corpus_path, "w")
        corpus_file.write(corpus_string)
        corpus_file.close()

        print("Corpus downloaded to", corpus_path)
    else:
        print("Corpus already downloaded.")



def preprocess_corpus_if_necessary():
    """
    Preprocesses the corpus either if it has not been done before or if it is
    forced.
    """

    if not os.path.exists(preprocessed_corpus_path) or preprocess_anyway == True:
        print("Preprocessing corpus...")

        # Opening the file.
        corpus_file = open(corpus_path, "r")
        corpus_string = corpus_file.read()

        # Getting the vocabulary.
        print("Tokenizing...")
        corpus_tokens = word_tokenize(corpus_string)
        print("Number of tokens:", len(corpus_tokens))
        print("Building vocabulary...")
        word_counter = Counter()
        word_counter.update(corpus_tokens)
        print("Length of vocabulary before pruning:", len(word_counter))
        vocabulary = [key for key, value in word_counter.most_common(most_common_words_number)]
        print("Length of vocabulary after pruning:", len(vocabulary))

        # Converting to indices.
        print("Index-encoding...")
        indices = encode_sequence(corpus_tokens, vocabulary)
        print("Number of indices:", len(indices))

        # Saving.
        print("Saving file...")
        pickle.dump((indices, vocabulary), open(preprocessed_corpus_path, "wb"))
    else:
        print("Corpus already preprocessed.")


def train_neural_network():
    """
    Trains the corpus either if it has not been done before or if it is
    forced.
    """

    if not os.path.exists(model_path) or train_anyway == True:

        # Loading index-encoded corpus and vocabulary.
        indices, vocabulary = pickle.load(open(preprocessed_corpus_path, "rb"))

        # Get the dataset.
        print("Getting the dataset...")
        data_input, data_output = get_dataset(indices)
        data_output = utils.to_categorical(data_output, num_classes=len(vocabulary))

        # Creating the model.
        print("Creating model...")
        model = models.Sequential()
        model.add(layers.Embedding(len(vocabulary), hidden_size, input_length=sequence_length))
        model.add(layers.LSTM(hidden_size))
        model.add(layers.Dense(len(vocabulary)))
        model.add(layers.Activation('softmax'))
        model.summary()

        # Compining the model.
        print("Compiling model...")
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy']
        )

        # Training the model.
        print("Training model...")
        history = model.fit(
            data_input, data_output,
            epochs=epochs, batch_size=batch_size)
        model.save(model_path)
        plot_history(history)


def get_dataset(indices):
    """ Gets a full dataset of a defined size from the corpus. """

    print("Generating data set...")
    data_input = []
    data_output = []
    current_size = 0
    bar = progressbar.ProgressBar(max_value=dataset_size)
    while current_size < dataset_size:

        # Randomly retriev a sequence of tokens and the token right after it.
        random_index = random.randint(0, len(indices) - (sequence_length + 1))
        input_sequence = indices[random_index:random_index + sequence_length]
        output_sequence = indices[random_index + sequence_length]

        # Update arrays.
        data_input.append(input_sequence)
        data_output.append(output_sequence)

        # Next step.
        current_size += 1
        bar.update(current_size)
    bar.finish()

    # Done. Return NumPy-arrays.
    data_input = np.array(data_input)
    data_output = np.array(data_output)
    return (data_input, data_output)


def generate_texts():
    """ Generates a couple of random texts. """

    print("Generating texts...")

    # Getting all necessary data. That is the preprocessed corpus and the model.
    indices, vocabulary = pickle.load(open(preprocessed_corpus_path, "rb"))
    model = models.load_model(model_path)

    # Generate a couple of texts.
    for _ in range(10):

        # Get a random temperature for prediction.
        temperature = random.uniform(0.0, 1.0)
        print("Temperature:", temperature)

        # Get a random sample as seed sequence.
        random_index = random.randint(0, len(indices) - (generated_sequence_length))
        input_sequence = indices[random_index:random_index + sequence_length]

        # Generate the sequence by repeatedly predicting.
        generated_sequence = []
        generated_sequence.extend(input_sequence)
        while len(generated_sequence) < generated_sequence_length:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            predicted_index = get_index_from_prediction(prediction[0], temperature)
            generated_sequence.append(predicted_index)
            input_sequence = input_sequence[1:]
            input_sequence.append(predicted_index)

        # Convert the generated sequence to a string.
        text = decode_indices(generated_sequence, vocabulary)
        print(text)
        print("")


def get_index_from_prediction(prediction, temperature=0.0):
    """ Gets an index from a prediction. """

    # Zero temperature - use the argmax.
    if temperature == 0.0:
        return np.argmax(prediction)

    # Non-zero temperature - do some random magic.
    else:
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_prediction= np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        probabilities = np.random.multinomial(1, prediction, 1)
        return np.argmax(probabilities)


def encode_sequence(sequence, vocabulary):
    """ Encodes a sequence of tokens into a sequence of indices. """

    return [vocabulary.index(element) for element in sequence if element in vocabulary]


def decode_indices(indices, vocabulary):
    """ Decodes a sequence of indices and returns a string. """

    decoded_tokens = [vocabulary[index] for index in indices]
    if use_moses_detokenizer  == True:
        return detokenizer.detokenize(decoded_tokens, return_str=True)
    else:
        return " ".join(decoded_tokens)


def plot_history(history):
    """ Plots the history of a training. """

    print(history.history.keys())

    # Render the loss.
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("history_loss.png")
    plt.clf()

    # Render the accuracy.
    plt.plot(history.history['categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("history_accuracy.png")
    plt.clf()

    plt.show()


if __name__ == "__main__":
    main()
