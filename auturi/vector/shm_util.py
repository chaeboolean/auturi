ALIGN_BYTES = 64
from multiprocessing import shared_memory as shm


def align(dummy_arr):
    if isinstance(dummy_arr, int):
        return 32
    else:
        return ((dummy_arr.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES
