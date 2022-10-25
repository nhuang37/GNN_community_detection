import math
import random
import os
import numpy as np
import numpy.random as npr
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Amazon, Coauthor, Actor, SNAPDataset
from torch_geometric.utils import to_networkx, homophily
import time
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse.linalg.eigen.arpack import ArpackError
import pandas as pd
import seaborn as sns
import copy
import argparse
from collections import defaultdict

#reproducibility
np.random.seed(0)
torch.manual_seed(0)

def embed(A, Xouter, d=2, scale=False):
  '''
  Embed the graph/covariance matrix together
  '''
  try: #error may occur if d is too large
    evalues, evectors = eigsh(A, k=d)
  except ArpackError:
  	for i in range(1,30):
  	  try:
  	    evalues, evectors = eigsh(A, k=d-i)
  	  except ArpackError:
  	  	continue
  	  else:
  	  	print(f"successfully solve sparse SVD at k={d-i}")
  	  	break
  evaluesX, evectorsX = eigsh(Xouter, k=d)

  if scale:
    ASE = evectors[:, :d] * evalues[:d]
    covX = evectorsX[:, :d] * evaluesX[:d]
  else:
    ASE = evectors[:, :d]
    covX = evectorsX[:, :d]

  return torch.FloatTensor(np.concatenate((ASE, covX), axis=1))


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, hidden_dim, out_dim, n_layers=2, dropout=0.5):
    super().__init__()
    self.layers = nn.ModuleList()
    self.n_layers = n_layers
    if n_layers == 1:
        self.layers.append(nn.Linear(input_dim, out_dim))
    else:
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
    if self.n_layers > 1:
        self.prelu = nn.PReLU()
        #self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    self.reset_parameters()

  def reset_parameters(self):
    gain = nn.init.calculate_gain("relu")
    for layer in self.layers:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)


  def forward(self, x):
    for layer_id, layer in enumerate(self.layers):
        x = layer(x)
        if layer_id < self.n_layers - 1:
            x = self.dropout(self.prelu(x))
    return x

def FilterFunction(weights, S, x):
    '''
    weights is a list of length k, with each element of shape d_in x d_out
    S is N x N, sparse matrix
    x is N x d, d-dimensional feature (i.e., unique node ID in the featureless setting)
    '''    
    # Number of filter taps
    K = len(weights)
    
    # Number of output features
    F = weights[0].shape[1]
 
    # Number of input features
    G = weights[0].shape[0]
    
    # Number of nodes
    N = S.shape[0]

    # Create list to store graph diffused signals
    zs = [x]
    
    # Loop over the number of filter taps / different degree of S
    for k in range(1, K):        
        # diffusion step, S^k*x
        x = torch.spmm(S, x) #torch.matmul(x, S) -- slow
        # append the S^k*x in the list z
        zs.append(x)
    
    # sum up
    out = [z @ weight for z, weight in zip(zs, weights)]
    out = torch.stack(out)
    y = torch.sum(out, axis=0)
    return y
    

class Graph_Perception(nn.Module):
    '''
    k: the degree of polynomial (on the graph operator) 
    f_in: input feature dimension
    f_out: output feature dimension
    '''
    def __init__(self, k, f_in, f_out, nonlinear=True, dropout=0.5):
        super().__init__()
        self.k = k
        self.f_in = f_in
        self.f_out = f_out
        self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.f_in,self.f_out)) for k in range(self.k)])
        self.nonlinear_flag = nonlinear
        self.dropout = nn.Dropout(dropout)
        if nonlinear:
          #self.act = nn.ReLU()
          self.act = nn.PReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.f_in * self.k)
        for elem in self.weight:
          elem.data.uniform_(-stdv, stdv)

    def forward(self, x, gso):
        y = FilterFunction(self.weight, gso, x)
        if self.nonlinear_flag:
          return self.dropout(self.act(y))
        else:
          return self.dropout(y)  

class GNN(nn.Module):
    def __init__(self, k, input_dim, hidden_dim, out_dim, num_layer, nonlinear=True):
        super().__init__()  
        self.k = k
        self.layers = nn.ModuleList()
        assert num_layer > 1, "must have at least 1 input layer and 1 output layer"
        # input layer
        self.layers.append(Graph_Perception(k, input_dim, hidden_dim, nonlinear=nonlinear))
        # hidden layers
        for i in range(num_layer - 2):
            self.layers.append(Graph_Perception(k, hidden_dim, hidden_dim, nonlinear=nonlinear))
        # output layer
        self.layers.append(Graph_Perception(1, hidden_dim, out_dim, nonlinear=False)) #output is a linear layer!

    def forward(self, x, gso):
      y = x
      for layer in self.layers:     
        y = layer(y, gso)
      return y

###Need to use full batch here
def train(model, feats, g, labels, loss_fcn, optimizer, train_mask):
    model.train()
    if g is None:
      logits = model(feats)
    else:
      logits = model(feats, g)
    loss = loss_fcn(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, feats, g, labels, train_nid, val_nid, test_nid):
    model.eval()
    if g is None:
      logits = model(feats)
    else:
      logits = model(feats, g)
    preds = torch.argmax(logits, dim=-1)
    train_res = (preds[train_nid] == labels[train_nid]).sum()/len(train_nid)
    val_res = (preds[val_nid] == labels[val_nid]).sum()/len(val_nid)
    test_res = (preds[test_nid] == labels[test_nid]).sum()/len(test_nid)
    return train_res, val_res, test_res

def run_model(model, feats, g, labels,  train_nid, val_nid, test_nid,  
              lr, tol=1e-2, weight_decay=0, num_epochs=200, eval_every=1, verbose=False):
    #optim
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    prev_loss = 1e6

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        loss = train(model, feats, g, labels, loss_fcn, optimizer, train_nid)

        if epoch % eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, g, labels, train_nid, val_nid, test_nid)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(*acc)
            if verbose:
              print(log)
            if acc[1] > best_val:
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]
        if abs(prev_loss - loss) < tol:
        	#print("early-stopped!")
        	break
        prev_loss = loss

    if verbose:
      print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))
    
    return best_val, best_test

def run_exp(data, g, emb_dim, device='cuda:0', eval_every = 1, k=2, depth=2, hidden_dim=128,
            act=False, lr=1e-2, n_epochs = 200, tol=1e-2, verbose=False, GNN_flag=True, MC_runs=5, norm_flag=True):
  '''
  data: pytorch geometric data class
  g: graph adjacency. If norm is True, use normalized graph adjacency
  emb_dim: embedding dimension
  '''
  results = []
  loss_fcn = torch.nn.CrossEntropyLoss()

  out_feats = len(torch.unique(data.y))
  if norm_flag:
    degs = g.sum(axis=0).clip(min=1)
    norm = np.power(degs, -0.5)
    g = np.diag(norm) @ g @ np.diag(norm)
    print(f"normalizing, deg_max={degs.max()}, deg_mean={degs.mean()}")

  g_tensor = torch.FloatTensor(g).to_sparse() 
  g_tensor = g_tensor.to(device)
  #print(g_tensor)
  if GNN_flag:
  #print(depth)
    feats = data.x
  else:
    Xouter = data.x.numpy() @ data.x.numpy().T
    #print(f'A: {A.shape}, Cov(X): {Xouter.shape}')
    feats = embed(g, Xouter, d=emb_dim)

  feats = feats.to(device)
  labels = data.y.to(device)

  if data.train_mask.dim() == 1:
    runs = 1
  else: 
    runs = data.train_mask.shape[1]
  #print('Total runs=', runs)
  #results
  for run in range(runs):
    if data.train_mask.dim() == 1:
      train_mask = data.train_mask
      val_mask = data.val_mask
      test_mask = data.test_mask 
    else:      
      train_mask = data.train_mask[:,run]
      val_mask = data.val_mask[:,run]
      test_mask = data.test_mask[:, run]
    nids = torch.arange(data.num_nodes)
    train_nid = nids[train_mask]
    val_nid = nids[val_mask]
    test_nid = nids[test_mask]
    acc_test = np.zeros(MC_runs)

    for MC in range(MC_runs):
      if GNN_flag:
        model = GNN(k=k, input_dim=feats.shape[1], hidden_dim=hidden_dim, out_dim=out_feats, num_layer=depth, nonlinear=act) 
      else:
        model = MLP(input_dim=feats.shape[1], hidden_dim=2*feats.shape[1], out_dim=out_feats)
        g_tensor = None
      
      model = model.to(device)
      best_val, best_test = run_model(model, feats, g_tensor, labels, train_nid, val_nid, test_nid, lr)
      acc_test[MC] = best_test.cpu().numpy()

    results.append(acc_test.mean())

  return results

def train_test_split(data):
    #first create a new in-memory dataset, and then add the train/val/test masks
    #same split as: https://github.com/cf020031308/3ference/blob/master/main.py
    data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y,
              train_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool),
              test_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool),
              val_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool))
    n_nodes = data.num_nodes
    val_num = test_num = int(n_nodes * 0.2)

    for run in range(10):
        torch.manual_seed(run)
        idx = torch.randperm(n_nodes)
        data_new.train_mask[idx[(val_num + test_num):], run] = True
        data_new.val_mask[idx[:val_num], run] = True
        data_new.test_mask[idx[val_num:(val_num + test_num)], run] = True
    return data_new

def delete_edge(A, sparsify_pct=10, seed=0):
  '''
  Input: binary adjacency matrix (symmetric, hollow)
  Output: modified adjacency matrix with sparsify_pct percentage of edges removed
  '''
  np.random.seed(seed)

  assert np.diag(A).sum() == 0, "not hollow!"
  assert np.allclose(A, A.T), "not symmetric!"
  assert ((A==0) | (A==1)).all(), "not binary!"

  E = A.sum()
  num_edges = int( (E/2) * (sparsify_pct/100))
  x_list, y_list = np.where(np.triu(A) == 1) #only take the upper triangular matrix to avoid dups
  edge_list = [(x,y) for (x,y) in zip(x_list, y_list)]
  A_del = copy.deepcopy(A)
  sample_edges = random.sample(edge_list, k=num_edges)
  for (i,j) in sample_edges:
    A_del[i,j] = 0
    A_del[j,i] = 0
  A_del += np.eye(A.shape[0]) #add self-loop
  return A_del

def delete_edge_local(A, knn=5, seed=0):
  '''
  Input: binary adjacency matrix (symmetric, hollow) A; knn - the number of local neighbors to sample
  Output: modified adjacency matrix with all nodes having degree min(dv, knn)
  '''
  np.random.seed(seed)

  assert np.diag(A).sum() == 0, "not hollow!"
  assert np.allclose(A, A.T), "not symmetric!"
  assert ((A==0) | (A==1)).all(), "not binary!"


  ###pretend A is a directed graph and sample per node
  n = A.shape[0]
  A_del = np.zeros((n,n))
  D = A.sum(axis=0)
  for i in range(n):
    if D[i] <= knn:
      pass #keep all the original edges
    else:
      neighbor_list = np.where(A[i,:] == 1)[0].tolist()
      neighbor_sample = random.sample(neighbor_list, knn)
      for j in neighbor_sample:
        A_del[i,j] = 1 #directed graph add edge per node's adjacency list
  ###turn the directed (non-symmetric) graph into a undirected (symmetric) one
  for i in range(n):
    for j in range(n):
      if A_del[i,j] != A_del[j,i]:
        A_del[i,j] = A_del[j,i] = 1
  A_del += np.eye(n)
  edge_drop_ratio = 100*(1 - A_del.sum() / A.sum())
  print(f'new graph drops {edge_drop_ratio:.2f} percentage of edges')
  ###checks
  assert np.allclose(A_del, A_del.T), "new graph not symmetric!"
  assert ((A_del==0) | (A_del==1)).all(), "new graph not binary!"
  return A_del

def run_data(data, undirected=True, eval_every=1, device='cuda:0', k=2, 
	         sparsify=0, depth=2, spectral_only=False, gnn_only=False, sp_dim=200, norm_flag=True, delete_type='global', num_local_edges=5):
  '''
  data: pytorch geometric data
  undirected: If true, add reverse edge to symmetrize the original graph
  '''
  #preprocess data
  nx_graph = to_networkx(data,to_undirected=undirected, remove_self_loops=True)
  A = nx.to_numpy_array(nx_graph)
  
  #tuple(emb_dim, GNN_flag, nonlinearity_flag)
  if spectral_only:
    name_dict = {f'SE({sp_dim})': (sp_dim, False, False)}
  elif gnn_only:
    name_dict = {'GNN(lin)':(None, True, False), 'GNN(non)':(None, True, True)}
  else:
    name_dict = {'SE(150)':(150, False,False),'SE(200)':(200, False,False),
               'GNN(lin)':(None, True, False), 'GNN(non)':(None, True, True)}


  if sparsify > 0 or delete_type=='local': #sparsify the input graph by dropping edges randomly
    results = defaultdict(list)
      #repeat for 10 runs
    for sp_seed in range(10):
      print(f"running trial {sp_seed}")
      if delete_type == 'global':
        A_sp = delete_edge(A, sparsify_pct=sparsify, seed=sp_seed)
      else:
        A_sp = delete_edge_local(A, knn=num_local_edges, seed=sp_seed)

      for name, flag in name_dict.items():
        print(f"running model {name}")
        emb_dim, GNN_flag, act_flag = flag
        accs = run_exp(data, A_sp, emb_dim, act=act_flag, GNN_flag=GNN_flag, depth=depth, k=k, norm_flag=norm_flag)
        results[name].append(accs)
        mean_acc = sum(accs) / len(accs)
        print(f"finish run={sp_seed} for model={name}, mean_acc={mean_acc}")
  else: #original graph
    results = {}
    A += np.eye(A.shape[0])
    for name, flag in name_dict.items():
      print(f"running model {name}")
      emb_dim, GNN_flag, act_flag = flag
      accs = run_exp(data, A, emb_dim, act=act_flag, GNN_flag=GNN_flag, depth=depth, k=k, norm_flag=norm_flag)
      results[name] = accs
      mean_acc = sum(accs) / len(accs)
      print(f"finish for model={name}, mean_acc={mean_acc}")

  return results

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    results_all = {}

    if args.datasets == 'wiki':
      datafunction = WikipediaNetwork
      names = ['chameleon'] #['chameleon', 'squirrel']
    elif args.datasets == 'planetoid':
    	datafunction = Planetoid
    	names = ['Cora', 'CiteSeer']
    elif args.datasets == 'webkb':
    	datafunction = WebKB
    	names = ['Cornell', 'Texas', 'Wisconsin']
    elif args.datasets == 'amazon':
    	datafunction = Amazon
    	names = ['photo'] #['computers', 'photo']
    elif args.datasets == 'actor':
      datafunction = Actor
      names = ['Actor']
    elif args.datasets == 'coauthor':
      datafunction = Coauthor
      names = ['CS']
    else:
      raise NotImplementedError
    
    for name in names:
      print(f"dowloading {name}")
      if args.datasets == 'planetoid':
        data_all = datafunction(root=args.data_path, name=name, split=args.data_split) 
      elif args.datasets == 'actor':
        data_all = datafunction(root=args.data_path)
      else: 
        data_all = datafunction(root=args.data_path, name=name)

      data = data_all[0]
      if args.datasets in ['amazon', 'coauthor']: #create 60/20/20 split for 10 runs
      	data = train_test_split(data)
      

      results_all[name] = run_data(data, eval_every=args.eval_every, device=device, k=args.poly_degree, 
      	                           sparsify=args.sparsify_pct, depth=args.depth, spectral_only=args.spectral_only, gnn_only=args.gnn_only, 
                                   sp_dim = args.spectral_dim, norm_flag=args.norm, delete_type=args.del_type, num_local_edges=args.num_edges)


    specs = f'data={args.datasets}_pct={args.sparsify_pct}_degree={args.poly_degree}_SPonly={args.spectral_only}_GNNonly={args.gnn_only}_norm={args.norm}_type={args.del_type}_edges={args.num_edges}'
    file_name = os.path.join(args.result_path, specs)
    pickle.dump(results_all, open(file_name, "wb" ))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerEmbed")
    parser.add_argument("--datasets", type=str, default="wiki", help="datasets: wiki / planetoid / webkb / amazon / actor / coauthor")
    parser.add_argument("--data_split", type=str, default="geom-gcn", help="dataset split type")
    parser.add_argument("--data_path", type=str, default="./dataset/", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./result/", help="dataset folder path")
    parser.add_argument("--eval_every", type=int, default=1,help="evaluation every k epochs")
    parser.add_argument("--depth",  type=int, default=2, help="GNN depth")
    parser.add_argument("--spectral_dim",  type=int, default=200, help="spectral_dim")

    parser.add_argument("--sparsify_pct",  type=int, default=0, help="sparsify the graph by keeping pct edges ")
    parser.add_argument("--poly_degree",  type=int, default=2, help="polynomial filter degree")
    parser.add_argument("--spectral_only",  action='store_true', help="default false; if true, only run spectral")
    parser.add_argument("--gnn_only",  action='store_true', help="default false; if true, only run gnns")
    parser.add_argument("--norm", action='store_true', help="default true: use normalized adjacency; if false, use un-normalized adj")
    parser.add_argument("--del_type", type=str, default="global", help="drop edge type: global - uniform sampling; local - local neighbor sampling")    
    parser.add_argument("--num_edges",  type=int, default=5, help="when choosing del_type == local, the number of local neighbors to keep")

    args = parser.parse_args()

    print(args)
    main(args)

