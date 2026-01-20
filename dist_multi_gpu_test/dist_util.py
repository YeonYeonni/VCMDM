"""
Helpers for distributed training.
"""

import os
import socket

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    """
    global used_device
    
    if dist.is_initialized():
        return
    
    # DDP를 사용할 경우
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        used_device = local_rank
        
        # NCCL backend for GPU training
        backend = "nccl" if th.cuda.is_available() else "gloo"
        
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        th.cuda.set_device(local_rank)
        print(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        # Single GPU mode
        used_device = device


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device >= 0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def get_rank():
    """
    Get the rank of the current process.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """
    Get the total number of processes.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    """
    Check if this is the main process (rank 0).
    """
    return get_rank() == 0


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # Map to correct device
    map_location = kwargs.get('map_location', dev())
    return th.load(path, map_location=map_location, **{k: v for k, v in kwargs.items() if k != 'map_location'})


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_initialized():
        for p in params:
            with th.no_grad():
                dist.broadcast(p, 0)


def cleanup():
    """
    Cleanup the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

# """
# Helpers for distributed training.
# """

# import socket

# import torch as th
# import torch.distributed as dist

# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).
# GPUS_PER_NODE = 8

# SETUP_RETRY_COUNT = 3

# used_device = 0

# def setup_dist(device=0):
#     """
#     Setup a distributed process group.
#     """
#     global used_device
#     used_device = device
#     if dist.is_initialized():
#         return
#     # os.environ["CUDA_VISIBLE_DEVICES"] = str(device) # f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

#     # comm = MPI.COMM_WORLD
#     # backend = "gloo" if not th.cuda.is_available() else "nccl"

#     # if backend == "gloo":
#     #     hostname = "localhost"
#     # else:
#     #     hostname = socket.gethostbyname(socket.getfqdn())
#     # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     # os.environ["RANK"] = str(comm.rank)
#     # os.environ["WORLD_SIZE"] = str(comm.size)

#     # port = comm.bcast(_find_free_port(), root=used_device)
#     # os.environ["MASTER_PORT"] = str(port)
#     # dist.init_process_group(backend=backend, init_method="env://")


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     global used_device
#     if th.cuda.is_available() and used_device>=0:
#         return th.device(f"cuda:{used_device}")
#     return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     return th.load(path, **kwargs)


# def sync_params(params):
#     """
#     Synchronize a sequence of Tensors across ranks from rank 0.
#     """
#     for p in params:
#         with th.no_grad():
#             dist.broadcast(p, 0)


# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()
