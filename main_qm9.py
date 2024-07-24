from qm9 import dataset
from qm9.simple_egnn import EGNN
import torch
import os
import numpy as np
import copy
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import json
from torch import linalg as LA
import time


def reg_spectral_ours(model, alpha):
    norms_per_egnn_layer = dict()
    for i in range(0, 8):  # 8 is just an arbitrary large enough number
        aux = [LA.matrix_norm(param, ord=2) for name, param in model.named_parameters() if
               ((f'gcl_{i}' in name) and ('.weight' in name))]
        if len(aux) > 0:
            norms_per_egnn_layer[i] = aux
    l_max_egnn = len(norms_per_egnn_layer)

    out_norms = [LA.matrix_norm(param, ord=2) for name, param in model.named_parameters() if
                 ((f'graph_dec.' in name) and ('.weight' in name))]

    P1 = torch.zeros(1).to(model.device)
    for l in range(l_max_egnn):
        aux = (l_max_egnn - l + 1)
        s = 0
        for n in norms_per_egnn_layer[l]:
            s += torch.log(torch.max(torch.tensor(1.).to(model.device), n))
        P1 += (aux * s).to(model.device)
    P2 = torch.sum(
        torch.log(torch.maximum(torch.tensor(1.0).to(model.device), torch.tensor(out_norms).to(model.device))))
    P = P1 + P2
    return alpha * P


def reg_spectral(model, alpha):
    norm = 0
    count = 0
    for name, param in model.named_parameters():
        if ('.weight' in name):
            norm += LA.matrix_norm(param, ord=2)
            count += 1
    return (alpha * norm / count).to(model.device)


parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--logdir', type=str, default='results', help='Log directory')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N')
parser.add_argument('--test_interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=8, metavar='N')  # number of hidden features
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')

parser.add_argument('--no_normalization', dest="normalize", action='store_false')
parser.add_argument('--squared_dist', action='store_true', default=False)

parser.add_argument("--regularizer", choices=["none", "spectral", "ours", "wd"], default="none")
parser.add_argument("--lambda_reg", type=float, default=1e-10)

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

args.logdir = f'{args.logdir}/{args.property}/{args.n_layers}_layers/{args.nf}_hidden/' \
              f'norm_{args.normalize}/sdist_{args.squared_dist}/{args.regularizer}/lambda_{args.lambda_reg}'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/'):
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)

model = EGNN(in_node_nf=15, hidden_nf=args.nf, in_edge_nf=0, device=device,
             n_layers=args.n_layers, squared_dist=args.squared_dist, normalize=args.normalize)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_reg * int(args.regularizer == 'wd'))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l2 = nn.MSELoss()
# loss_l1 = nn.L1Loss()

if args.regularizer == 'spectral':
    reg = reg_spectral
elif args.regularizer == 'ours':
    reg = reg_spectral_ours


def train(epoch, loader, partition='train'):
    res = {'loss': 0, 'counter': 0, 'loss_arr': []}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[args.property].to(device, dtype)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        if partition == 'train':
            loss = 0.5 * loss_l2(pred, (label - meann) / mad)
            #            loss = loss_l1(pred, (label - meann) / mad)
            if args.regularizer == 'spectral' or args.regularizer == 'ours':
                loss = loss + reg(model, args.lambda_reg)
            loss.backward()
            optimizer.step()
        else:
            loss = torch.minimum(torch.tensor(1.0), 0.5 * ((mad * pred + meann) - label) ** 2).mean()
        #            loss = 0.5*loss_l2(pred, (label - meann) / mad)
        #            loss = loss_l1(mad * pred + meann, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

    #        prefix = ""
    #        if partition != 'train':
    #            prefix = ">> %s \t" % partition
    #        if i % args.log_interval == 0:
    #            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter']


models = []
train_losses, train_losses_, val_losses, test_losses = [], [], [], []
test_epochs = []
train_times = []

for epoch in range(0, args.epochs):
    start = time.time()
    train_loss = train(epoch, dataloaders['train'], partition='train')
    end = time.time()
    train_times.append(end - start)

    lr_scheduler.step()
    train_losses.append(train_loss)

    with torch.no_grad():
        if epoch % args.test_interval == 0:
            test_epochs.append(epoch)
            train_loss_ = train(epoch, dataloaders['train'], partition='train_')
            val_loss = train(epoch, dataloaders['valid'], partition='valid')
            test_loss = train(epoch, dataloaders['test'], partition='test')
            print(f"Epoch {epoch}, train loss: {train_loss_:.4f}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}")

            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_losses_.append(train_loss_)

#        if (epoch % args.save_interval == 0) or (epoch == args.epochs-1):
#            torch.save(model.state_dict(), f'{args.logdir}/models/egnn_{args.seed}_epoch_{epoch}.model')

# torch.save(models, f'{args.logdir}/models/egnn_{args.seed}.models')
results = {
    'train_times': torch.tensor(train_times),
    'test_epochs': torch.tensor(test_epochs),
    'train_losses': torch.tensor(train_losses),
    'train_losses_': torch.tensor(train_losses_),
    'val_losses': torch.tensor(val_losses),
    'test_losses': torch.tensor(test_losses)
}
torch.save(results, f'{args.logdir}/egnn_{args.seed}.results')
