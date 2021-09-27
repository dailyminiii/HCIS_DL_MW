import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import mean_absolute_error

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_filters, filter_size, batch_size, dropout, use_bn, window_len):

        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.batch_size = batch_size

        self.dropout = dropout
        self.use_bn = use_bn
        self.window_len = window_len

        # Layers 2-5
        self.conv1 = nn.Conv1d(input_dim, num_filters, filter_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv4 = nn.Conv1d(num_filters, num_filters, filter_size)

        # Layers 6-7 - each dense layer has LSTM cells
        self.lstm1 = nn.LSTM(num_filters, hidden_dim, num_layers)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, num_layers)

        # Layer 9 - prepare for softmax
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Layer 1 - flatten (see -1)
        x = x.view(-1, self.input_dim, self.window_len)

        # Layers 2-5 - RELU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Layers 5 and 6 - flatten
        x = x.view(1, -1, self.num_filters)

        # Layers 6-8 - hidden layers
        x, self.hidden = self.lstm1(x, self.hidden)
        x, self.hidden = self.lstm2(x, self.hidden)
        x, self.hidden = self.lstm3(x, self.hidden)
        x, self.hidden = self.lstm4(x, self.hidden)
        x, self.hidden = self.lstm5(x, self.hidden)

        # Layers 8 - flatten, fully connected for softmax. Not sure what dropout does here
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)

        # View flattened output layer
        out = x.view(self.batch_size, -1, self.output_dim)[:, -1, :]

        return out

    def init_hidden(self):
        '''
        Initializes hidden state

        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM
        '''
        weight = next(self.parameters()).data

        # changed this from batch_size to 3*batch_size
        hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_(),
                 weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_())

        return hidden


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):

        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # print('lstm_out shape: ',lstm_out.shape)
        # print('lstm_out.view(self.batch_size, -1) shape: ', lstm_out.view(self.batch_size, -1).shape)
        out = self.fc(lstm_out.view(self.batch_size, -1))
        # print('out shape: ', out.shape)
        #out = self.fc(lstm_out.squeeze())
        return out


class Conv1D(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, num_filters, filter_size, batch_size, dropout, use_bn, window_len):

        super(Conv1D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.window_len = window_len

        self.dropout = dropout
        self.use_bn = use_bn

        # Layers 2-5
        self.conv1 = nn.Conv1d(input_dim, num_filters, filter_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv4 = nn.Conv1d(num_filters, num_filters, filter_size)

        # Layer 9 - prepare for softmax
        self.fc = nn.Linear(num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Layer 1 - flatten (see -1)
        x = x.view(-1, self.input_dim, self.window_len)
        # print(x.shape)

        # Layers 2-5 - RELU
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)

        # Layers 5 and 6 - flatten
        x = x.view(1, -1, self.num_filters)
        # print(x.shape)

        # Layers 8 - flatten, fully connected for softmax. Not sure what dropout does here
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)

        # View flattened output layer
        out = x.view(self.batch_size, -1, self.output_dim)[:, -1, :]
        # print(x.shape)

        return out

def metric(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    return mae*100


def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    #정확도 출력하기
    total = 0
    correct = 0
    train_loss = 0.0

    for i, (X, y) in enumerate(trainloader):

        # print(i, 'Processing Train Data ...')
        # print('X1 shape: ', X.shape)
        # print('y1 shape: ', y.shape)

        X = X.transpose(0, 1).float().to(args.device)
        # print('X2 shape: ', X.shape)
        y_true = y[:, 0].long().to(args.device)
        # print('y_true shape: ', y_true.shape)
        # print('y_true: ', y_true)

        model.zero_grad()
        optimizer.zero_grad()
        if args.model == 'ConvLSTM' or args.model == 'LSTM':
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        y_pred = model(X)

        # print('y_pred shape: ', y_pred.shape)
        # print('y_pred: ', y_pred)
        # yy = y_pred.squeeze()
        # print('yy: ', yy.shape)
        # print('yy: ', yy)
        loss = loss_fn(y_pred, y_true.view(-1))
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(y_pred.data, 1)
        # print('yyy', y_pred)
        train_loss += loss.item()
        # train_acc += (y_pred == y_true).sum()
        # train_acc += metric(y_pred, y_true)[0]
        total += y.size(0)

        correct += (y_pred == y_true).sum()
        correct = float(correct)

    '''
    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)
    train_acc = float(train_acc)
    '''
    train_loss = train_loss / len(trainloader)
    train_acc = 100*correct/total

    return model, train_loss, train_acc

def validate(model, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()

    #정확도 출력하기

    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

            # print(i, 'Processing Validation Data ...')

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, 0].long().to(args.device)
            if args.model == 'ConvLSTM' or args.model == 'LSTM':
                model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            loss = loss_fn(y_pred, y_true.view(-1))

            _, y_pred = torch.max(y_pred.data, 1)
            val_loss += loss.item()

            total += y.size(0)

            correct += (y_pred == y_true).sum()
            correct = float(correct)

    val_loss = val_loss / len(valloader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

def test(model, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):

            # print(i, 'Processing Test Data ...')

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, 0].long().to(args.device)
            if args.model == 'ConvLSTM' or args.model == 'LSTM':
                model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            _, y_pred = torch.max(y_pred.data, 1)

            total += y.size(0)

            correct += (y_pred == y_true).sum()
            correct = float(correct)

    test_acc = 100 * correct / total

    return test_acc

def experiment(partition, args):

    if args.model == 'ConvLSTM':
        model = ConvLSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.n_filters, args.filter_size, args.batch_size,
                         args.dropout, args.use_bn, args.str_len)
    elif args.model == 'LSTM':
        model = LSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size, args.dropout, args.use_bn)

    elif args.model == 'Conv1D':
        model = Conv1D(args.input_dim, args.y_frames, args.n_layers, args.n_filters, args.filter_size, args.batch_size, args.dropout, args.use_bn, args.str_len)
    else:
        raise ValueError('In-valid model choice')

    model.to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        # print('Start training ... ')
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args)
        # print('Start validation ... ')
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()

        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #

        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(epoch+1, train_acc, val_acc, train_loss, val_loss, te - ts))

    test_acc = test(model, partition, args)
    print('##### test_acc', test_acc, '#####')


    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    return vars(args), result
