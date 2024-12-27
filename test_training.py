import numpy as np

def generate_training_data(corpus):
    # Membuat vocabulary
    words = []
    for word_list in corpus:
        for word in word_list:
            if word not in words:
                words.append(word)
    vocab_size = len(words)

    # Membuat one-hot encoding
    word2int = {}
    int2word = {}
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    # Membuat data training
    data = []
    for sentence in corpus:
        for i, word in enumerate(sentence):
            context = [sentence[i - 1], sentence[(i + 1) % len(sentence)]]
            label = word
            data.append((context, label))

    return data, word2int, int2word

corpus = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
          ['this', 'is', 'the', 'second', 'sentence'],
          ['yet', 'another', 'sentence'],
          ['one', 'more', 'sentence'],
          ['and', 'the', 'final', 'sentence']]

data, word2int, int2word = generate_training_data(corpus)
vocab_size = len(word2int)
one_hot_encoding = np.eye(vocab_size)
# print(one_hot_encoding[word2int['word2vec']])
# exit()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward(context_words, target_word, hidden_weights, output_weights):
    # Menghitung hidden layer
    context_vectors = []
    for context_word in context_words:
        context_vectors.append(one_hot_encoding[word2int[context_word]])
    context_vectors = np.array(context_vectors)
    hidden_layer = np.mean(context_vectors, axis=0)

    # Menghitung output layer
    output_layer = softmax(np.dot(hidden_layer, output_weights))
    # print(output_layer)
    # print(target_word)
    # exit()

    # Menghitung loss dan gradient
    loss = -np.log(output_layer[word2int[target_word]])
    dl_dz = output_layer.copy()
    dl_dz[word2int[target_word]] -= 1.0

    # Menghitung gradient output layer dan hidden layer
    dl_dw_out = np.outer(hidden_layer, dl_dz)
    dl_dh = np.dot(output_weights, dl_dz)
    
    # Menghitung gradient hidden layer dan input layer
    dl_dw_in = np.zeros_like(hidden_weights)
    for i in range(len(context_words)):
        dl_dw_in += np.outer(context_vectors[i], dl_dh)

    return loss, dl_dw_in, dl_dw_out

def train(data, epochs, learning_rate):
    # Inisialisasi bobot
    hidden_size = 14
    hidden_weights = np.random.uniform(-1, 1, (vocab_size, hidden_size))
    output_weights = np.random.uniform(-1, 1, (hidden_size, vocab_size))

    # Training model
    for epoch in range(epochs):
        epoch_loss = 0.0
        for context_words, target_word in data:
            loss, dl_dw_in, dl_dw_out = forward(context_words,
                                                 target_word,
                                                 hidden_weights,
                                                 output_weights)
            epoch_loss += loss
            
            # Update bobot
            hidden_weights -= learning_rate * dl_dw_in
            output_weights -= learning_rate * dl_dw_out
            
        print(f'Epoch {epoch+1}, loss={epoch_loss:.4f}')

train(data=data,
      epochs=100,
      learning_rate=0.05)
