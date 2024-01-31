import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch.nn as nn
from typing import Optional, List
from torch.utils.data import Dataset
from torchvision import datasets
from math import ceil


rng = np.random.default_rng(0)


class customDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        self.n_samples = self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx]

    def __len__(self):
        return self.n_samples


class STLGenerator(object):
    def __init__(self, datapath):
        self.filenamesx, self.filenamesy = self.getXfiles(datapath)
 
    def __getitem__(self, idx):
        filenamex = self.filenamesx[idx]
        X = np.load(filenamex)
        X = torch.from_numpy(X.astype(np.float32))
        if self.filenamesy:
            filenamey = self.filenamesy[idx]
            y = np.load(filenamey)
            y = torch.from_numpy(y.astype(np.float32))
        else:
            y=None
        return X, y

    def __len__(self):
        return len(self.filenamesx)


    @staticmethod
    def getXfiles(datapath):
        allxpath = []
        allypath = []
        for root, _, files in os.walk((os.path.normpath(datapath)), topdown=False):
            dir = root.split('/')[-1]
            if dir != 'random':
                for name in files:
                    if name.endswith('x.npy'):
                        path = os.path.join(root, name)
                        allxpath.append(path)
                    elif name.endswith('y.npy'):
                        path = os.path.join(root, name)
                        allypath.append(path)
        return allxpath, allypath


class customGenerator(object):
    def __init__(self, datapath):
        self.filenamesx = self.getXfiles(datapath)
 
    def __getitem__(self, idx):
        filenamex = self.filenamesx[idx]
        X = np.load(filenamex)
        X = torch.from_numpy(X.astype(np.float32))
        return X

    def __len__(self):
        return len(self.filenamesx)


    @staticmethod
    def getXfiles(datapath):
        allxpath = []
        for root, _, files in os.walk((os.path.normpath(datapath)), topdown=False):
            for name in files:
                if name.endswith('.npy'):
                    path = os.path.join(root, name)
                    allxpath.append(path)

        return allxpath


class customImageFolder(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(customImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



class Autoencoder(nn.Module):
    def __init__(
        self, 
        inputsize: int,
        dims: List[int]
        ):
        super(Autoencoder, self).__init__()

        self.inputsize = inputsize
        
        encmodules = []
        encmodules.append(nn.Linear(inputsize, dims[0]))
        for index in range(len(dims)-1):
            encmodules.append(nn.ReLU(True))
            encmodules.append(nn.Linear(dims[index], dims[index+1]))
        self.encoder = nn.Sequential(*encmodules)

        decmodules = []
        for index in range(len(dims) - 1, 0, -1):
            decmodules.append(nn.Linear(dims[index], dims[index-1]))
            decmodules.append(nn.ReLU(True))
        decmodules.append(nn.Linear(dims[0], inputsize))
        self.decoder = nn.Sequential(*decmodules)

        self.init_weights()

    def forward(
        self, 
        x
        ):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_encoder(
        self
        ):
        return self.encoder

    def init_weights(
        self
        ):
        def func(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.00)

        self.encoder.apply(func)
        self.decoder.apply(func)
        
        
class Clustering(nn.Module):
    def __init__(
        self, 
        n_clusters:int,
        input_shape:int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None
        ) -> None:
        super(Clustering, self).__init__()

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_shape = input_shape

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.input_shape, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.clustcenters = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
            inputs: the variable containing data, shape=(n_samples, n_features)
            output: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, axis=1) - self.clustcenters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
 

class DEC(nn.Module):
    def __init__(
        self, 
        dims: List[int],
        inputsize: int, 
        n_clusters: int):
        super(DEC, self).__init__()
        
        self.AE = Autoencoder(inputsize, dims)
        self.clustlayer = Clustering(n_clusters, dims[-1])

        self.model = nn.Sequential(
            self.AE.encoder,
            self.clustlayer
        )    
        
    def forward(self, inputs):
        X = self.model(inputs)
        return X

    
def train_autoencoder(
        model:torch.nn.Module, 
        X: torch.tensor, 
        batch_size: int=256,
        lr=1e-3,
        epochs: int=10,
        savepath: str = './save/models/', 
        device = 'cuda:0',
        savemodel: bool=True,
        verbose_every=None,
    ):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.AE.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = 0
        count = 0
        model.AE.train()  # Set model to training mode

        dataset = customDataset(X)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for batch in dataloader:
            X = batch
            X = X.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model.AE(X)
                train_loss = criterion(outputs, X)
                train_loss.backward()
                optimizer.step()

            loss += train_loss.item()
            count+=1

        # compute the epoch training loss
        if verbose_every is not None and epoch % verbose_every == 0:
            loss = loss / count
            print(f'{loss}')
            
    # if savemodel:
    #     if not os.path.exists(savepath):
    #         os.mkdir(savepath)
    #     torch.save(model.AE.state_dict(), os.path.join(savepath,'ae_weights.pth'))


def train_dec(
        dec:torch.nn.Module, 
        X:torch.tensor,
        y=None,
        n_clusters:int=10,
        batch_size:int=256, 
        n_iters:int=10000,
        lr=0.01,
        update_target_every:int=30,
        device:str='cuda:0',
        update_freq:int=20,
        tol:float=0.001,
        verbose:bool=False,
        init_centroids=True,
        every=10,
    ):
    
    # initialising the cluster centres
    with torch.no_grad():
        if init_centroids:
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            y_pred_last = kmeans.fit_predict(dec.AE.encoder(X.to(device)).clone().detach().cpu())

            clustcenters = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
            clustcenters = clustcenters.to(device)

            dec.state_dict()["clustlayer.clustcenters"].copy_(clustcenters)
        else:
            q = dec(X.to(device)).detach().cpu().numpy()
            y_pred_last = np.argmax(q, axis=-1)
        
    # defining the loss function and optimizer settings
    criterion = nn.KLDivLoss(reduction='batchmean')
    # optimizer = torch.optim.Adam(dec.model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(dec.model.parameters(), lr=lr, momentum=0.9)
    
    index_array = np.arange(X.shape[0])
    index = 0
    loss = 0
    count = 1
    n_batches = ceil(X.shape[0] / batch_size)
    dec.train()
    for i in range(n_iters):
        bn = 0
        
        # instead of in every iteration, we update the auxiliary target distribution in every update_target_every
        if i % update_target_every == 0:
            with torch.no_grad():
                q = dec(X.to(device))
                p = dec.clustlayer.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
            
        index = rng.integers(0, n_batches - 1)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            idx = index_array[index * batch_size: min((index + 1) * batch_size, X.shape[0])]

            trainx = X[idx]
            trainx = trainx.to(device)
            trainy = p[idx]
            trainy = trainy.to(device)

            outputs = dec(trainx)
            # index = index + 1 if (index + 1) * batch_size < X.shape[0] else 0

            train_loss = criterion(outputs.log(), trainy)

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            
        # if i % np.ceil(X.shape[0]/batch_size) == 0:
        if i % every == 0:
            with torch.no_grad():
                q = dec(X.to(device)).detach().cpu().numpy()
                y_pred = np.argmax(q, axis=-1)

                if verbose:
                    ari = adjusted_rand_score(y, y_pred)
                    # print(f'{loss / (i + 1):.8f} {ari:.8f}')
                    print(f'{i:05d}: {ari:.2f}')
                # acc = np.round(metrics.acc(clusters_true, y_pred.clone().detach().cpu().numpy()), 5)
                # nmi = np.round(metrics.nmi(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
                # ari = np.round(metrics.ari(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
                # print('Epoch %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (epochs / np.ceil(X.shape[0]/batch_size), acc, nmi, ari), ' ; loss=', np.round(loss/count, 5))

                # check stop criterion, when less than tol%of points change cluster assignment between two consecutive epochs.
                delta_label = np.sum(y_pred_last!= y_pred) / y_pred.shape[0]
                if tol is not None and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break 
                y_pred_last = y_pred.copy()
            
#         if update_freq != 0 and i % update_freq == 0:
#            torch.save(model.state_dict(), 'save/models/dec_weights_%s.pth'%(n_clusters))
    dec.eval() 
#     torch.save(model.state_dict(), 'save/models/dec_weights_%s.pth'%(n_clusters))
    
