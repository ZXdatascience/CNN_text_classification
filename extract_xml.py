import xml.etree.ElementTree as ET
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv2D
from keras import initializers
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Input, Embedding, Dropout, Activation
from keras.layers.merge import concatenate

BASE_DIR = 
EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.txt'
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
def extract(i):
    tree= ET.parse('/resources/data/XML/%s.xml' %i)
    meta_data= tree.findall("./head/meta")
    meta= {data.get("name"): data.get("content") for data in meta_data}
    general_desc= tree.findall(""".//identified-content/classifier[@type='general_descriptor']""")
    general_description= {general_desc[0].get('type'): [data.text for data in general_desc]}
    words= tree.findall("./head/pubdata")
    word_count= {"word_count": word.get("item-length") for word in words}
    headline= tree.findall("./body//hedline/hl1")
    title= {"headline": hl.text for hl in headline}
    byLine = tree.findall("./body//byline")
    byline = {"byline": [line.text for line in byLine]}
    author_ = tree.findall("./body//byline[@class='normalized_byline']")
    author = {"autor": auth.text for auth in author_}
    lead_para= tree.findall("./body//*block[@class='lead_paragraph']/p")
    lead_paragraph= {"lead_paragraph": "\n".join([data.text for data in lead_para])}
    ft= tree.findall("./body//*block[@class='full_text']/p")
    full_text= {"full_text": "\n".join([re.sub('(\n)( )+',' ',data.text) for data in ft])}
    result= {}
    for dictionary in [meta, general_description, word_count, title, byline, author, lead_paragraph, full_text]:
        result.update(dictionary)
    return result

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
########################################
## load text and transfer to sequences
########################################
alltext = []
while(1):
    dictionary = extract(i)
    text= text_to_wordlist(dictionary['lead_paragraph'])
    alltext.append(text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(alltext)

sequences = tokenizer.texts_to_sequences(alltext)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)
########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE)
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
convolution_layer1 = Conv2D(filters = num_filters, kernel_size = [KERNEL_SIZES[0], EMBEDDING_DIM], data_format = 'channels_first', padding = 'VALID', activation = 'relu', use_bias = True,	
                           bias_initializer = initializers.Constant(value=0.1), kernel_initializer = initializer.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
convolution_layer2 = Conv2D(filters = num_filters, kernel_size = [KERNEL_SIZES[1], EMBEDDING_DIM], data_format = 'channels_first', padding = 'VALID', activation = 'relu', use_bias = True,	
                           bias_initializer = initializers.Constant(value=0.1), kernel_initializer = initializer.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
convolution_layer3 = Conv2D(filters = num_filters, kernel_size = [KERNEL_SIZES[2], EMBEDDING_DIM], data_format = 'channels_first', padding = 'VALID', activation = 'relu', use_bias = True,	
                           bias_initializer = initializers.Constant(value=0.1), kernel_initializer = initializer.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
## the input shape of Conv2D
## (samples, channels, rows, cols)
pool_layer = MaxPooling2D(pool_size=(MAX_SEQUENCE_LENGTH-kernel_size+1,1), strides=(1,1), padding='VALID', data_format='channels_first')
## the input shape of MaxPooling2D
## (batch_size, channels, rows, cols)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
conv_seq1 = convolution_layer(embedded_sequences)
conv_seq2 = convolution_layer(embedded_sequences)
conv_seq3 = convolution_layer(embedded_sequences)

pooled1 = pool_layer(conv_seq1)
pooled2 = pool_layer(conv_seq2)
pooled3 = pool_layer(conv_seq3)

pooled = concatenate([pooled1, pooled2, pooled3])













