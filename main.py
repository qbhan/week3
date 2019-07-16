# from google.colab import auth
# auth.authenticate_user()
#
# from google.colab import drive
# drive.mount('/content/gdrive')
#
# with open('/content/gdrive/My Drive/foo.txt', 'w') as f:
#
#   f.write('Hello Google Drive!')
#
# !cat /content/gdrive/My\ Drive/foo.txt


# !apt-get update
# !apt-get install g++ openjdk-8-jdk python-dev python3-dev
#
# !pip3 install JPype1-py3
#
# !pip3 install konlpy
#
# !JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from konlpy.tag import Kkma, Okt
from konlpy.utils import pprint
import os
import pickle

import torch
import torch.nn as nn
from sympy.physics.quantum.circuitplot import matplotlib
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

print(torch.cuda.current_device())

gpu_num = 0;

use_cuda = torch.cuda.is_available()

kkma = Kkma()
twitter = Okt()
MAX_LENGTH = 80

path_name = "/content/gdrive/My Drive/ByungSim_V_2_0/save"
save_dir = os.path.join("data", path_name)

# make dict

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # count SOS and EOS and UNKWON

    def addSentence(self, sentence):
        for word in twitter.morphs(sentence):  # kkma.morphs(sentence): # 형태소 단위로 단어 자르기
            # print(word)
            self.addWord(word)

    #         for word in sentence.split(' '):
    #             self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.num_words = 2  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Turn a Unicode stirng to plain ASCII

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣 ^a-zA-Z.!? 0-9 \t]+')
    result = hangul.sub('', s)

    return result


def readText():
    print("Reading pairs...")
    letter = "/content/gdrive/My Drive/train_7891.txt"
    letter_path = os.path.join("data", letter)
    with open(letter_path, 'rb') as f:
        pairs = pickle.load(f)
    print("Num pairs : ", len(pairs))

    print("Counting words...")
    words = Lang('words')
    for i in range(len(pairs)):
        words.addSentence(pairs[i][0])
        words.addSentence(pairs[i][1])
    print("words length : ", words.n_words)
    return words, pairs


MIN_COUNT = 2  # Minimum word count threshold for trimming


def trimRareWords(words, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    words.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in twitter.morphs(input_sentence):
            if word not in words.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in twitter.morphs(output_sentence):
            if word not in words.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs

# Trim voc and pairs
# pairs = trimRareWords(voc, pairs, MIN_COUNT)

#   pairs = []
#   pairss = []
#   inputs = []
#   outputs = []

#   letter = "/content/gdrive/My Drive/new_pairs.txt"
#   letter_path = os.path.join("data", letter)


#   with open(letter_path, 'rb') as f:
#     pairs = pickle.load(f)

#   for pair in pairs:
# #     print(pair)
#     inputs.append(normalizeString(pair[0]))
#     outputs.append(normalizeString(pair[1]))
#     pairss.append([normalizeString(pair[0]), normalizeString(pair[1])])

#   print(len(inputs))
#   print(len(outputs))
#   print(len(pairss))

#   inp = Lang('input')
#   outp = Lang('output')

# #   print(inp.n_words)
# #   print(outp.n_words)

#   return inp, outp, pairss


words, pairs = readText()

pairs = trimRareWords(words, pairs, MIN_COUNT)

print(random.choice(pairs))

words.word2index['물건']


def indexesFromSentence(lang, sentence):
    ret = []
    for word in twitter.morphs(sentence):  # kkma.morphs(sentence):#sentence.split(' '):
        ret.append(lang.word2index[word])
    return ret


#      return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    #     print(indexes)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda(gpu_num)
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(words, pair[0])
    target_variable = variableFromSentence(words, pair[1])
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = F.relu(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda(gpu_num)
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), 1)
        #         attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]) , 1 )))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.relu(output)
        output, hidden = self.gru2(output, hidden)
        #         output = F.log_softmax(self.out(output[0]))
        output = F.log_softmax(self.out(output[0]), 1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda(gpu_num)
        else:
            return result


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda(gpu_num) if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda(gpu_num) if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda(gpu_num) if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    #   print(loss)

    #   return loss.data[0] / target_length
    return loss.item() / target_length


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m,s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, n_epochs,
               print_every, plot_every, learning_rate, loadFilename):
    start = time.time()

    plot_losses = []

    training_pairs = [variablesFromPair(pairs[i]) for i in range(len(pairs))]
    criterion = nn.NLLLoss()

    # initializations
    print('Initializing ...')
    start_epoch = 1
    if loadFilename:
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, n_epochs + 1):
        print('epoch : ', epoch)
        print_loss_total = 0
        plot_loss_total = 0
        for i in range(len(pairs)):
            training_pair = training_pairs[i - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                         criterion)
            print_loss_total += loss
            plot_loss_total += loss

            # print progress
            if (i + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%d : %.4f' % (i + 1, print_loss_avg))
            #                 print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100 , print_loss_avg))
            if (i + 1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        ##save checkpoint
        directory = os.path.join(save_dir, '{}_{}'.format(learning_rate, hidden_size))
        print("directory: ", directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("no directory -> make")
        torch.save({
            'epoch': epoch,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'words_dict': words.__dict__,
            #                     'embedding': embedding.state_dict()
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))
        print("saved..")

    showPlot(plot_losses)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(words, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda(gpu_num) if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda(gpu_num) if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(words.index2word[int(ni)])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda(gpu_num) if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# def evaluateInput(encoder, decoder):
#     input_sentence = ''
#     while(1):
#         try:
#             # Get input sentence
#             input_sentence = input('> ')
#             # Check if it is quit case
#             if input_sentence == 'q' or input_sentence == 'quit': break
#             # Normalize sentence
#             input_sentence = normalizeString(input_sentence)
#             # Evaluate sentence
#             output_words = evaluate(encoder, decoder, input_sentence, max_length=MAX_LENGTH)
#             # Format and print response sentence
#             output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
#             print('Bot:', ' '.join(output_words))

#         except KeyError:
#             print("Error: Encountered unknown word.")


hidden_size = 512

# set checkpoint to load from; set to None if starting from scratch
loadFilename = None
# checkpoint_epoch = 57
n_epochs = 50
print_every = 500
plot_every = 50
learning_rate = 0.005
# loadFilename = os.path.join(save_dir, '{}_{}'.format(learning_rate, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_epoch))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    #     embedding_sd = checkpoint['embedding']

    words.__dict__ = checkpoint['words_dict']

encoder1 = EncoderRNN(words.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, words.n_words, dropout_p=0.1)

if loadFilename:
    encoder1.load_state_dict(encoder_sd)
    attn_decoder1.load_state_dict(decoder_sd)

if use_cuda:
    encoder1 = encoder1.cuda(gpu_num)
    attn_decoder1 = attn_decoder1.cuda(gpu_num)

encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

trainIters(encoder1, attn_decoder1, encoder_optimizer, decoder_optimizer, n_epochs,
           print_every, plot_every, learning_rate, loadFilename)


evaluateRandomly(encoder1, attn_decoder1)

def evaluateInput(encoder, decoder):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words, decoder_attentions = evaluate(encoder, decoder , input_sentence)
#             print("--output-- : ", output_words)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

input_sentence = '안녕'

output_words, decoder_attentions = evaluate(encoder1, attn_decoder1, input_sentence)

output_words

evaluateInput(encoder1, attn_decoder1)