import os
N_THREADS = "8"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["NUMBA_NUM_THREADS"] = N_THREADS

import time
import torch
import random
import numpy as np
from numba import jit, prange, set_num_threads

torch.set_num_threads(int(N_THREADS))

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@jit(nopython=True, parallel=True)
def csc_reorder(indptr, indices, new_degs, new_indptr, map_new2old, map_old2new):
    num_nodes = new_degs.shape[0]
    num_edges = indices.shape[0]
    new_indices = np.zeros_like(indices, dtype=np.int64)
    for i in prange(num_nodes):
        tmp_edges = new_degs[i]
        offset_new = new_indptr[i]
        offset_old = indptr[map_new2old[i]]
        for j in prange(tmp_edges):
            new_indices[offset_new + j] = map_old2new[indices[offset_old + j]]
    return new_indices


def construct_and_reorder(num_nodes=2e7, num_edges=2e8):
    num_nodes = int(num_nodes)
    num_edges = int(num_edges)
    print(f"rand graph with {num_nodes/1e6:.2f} million nodes and {num_edges/1e9:.2f} billion edges")

    set_seeds(0)
    tic = time.time()
    # construct indptr
    degs = torch.rand((num_nodes,))
    degs = degs / degs.sum()
    degs = (degs * num_edges).long() # conversion is rounded off
    degs[0] += num_edges - degs.sum()
    indptr = torch.zeros((num_nodes+1, ), dtype=torch.long)
    torch.cumsum(degs, 0, out=indptr[1:])
    # construct indices
    indices = torch.randint(num_nodes, (num_edges,), dtype=torch.long)
    # construct others
    adj_order = torch.randperm(num_nodes)
    mmap = np.zeros(num_nodes, dtype=np.int64)
    mmap[adj_order.numpy()] = np.arange(num_nodes)
    new_degs = degs[adj_order].numpy()
    new_indptr = np.zeros_like(indptr.numpy())
    new_indptr[1:] = new_degs.cumsum()
    # reorder
    new_indices = csc_reorder(indptr.numpy(), indices.numpy(), new_degs, new_indptr, adj_order.numpy(), mmap)

    dur = time.time() - tic
    print(f"time elapsed : {dur:.4f}s")
    return dur

if __name__ ==  '__main__':
    for _ in range(10):
        construct_and_reorder()
