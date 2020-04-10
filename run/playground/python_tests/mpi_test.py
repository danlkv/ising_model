from mpi4py import MPI
import platform

w = MPI.COMM_WORLD
comm = MPI.Comm
mprank = comm.Get_rank(w)

print(f'I am `{platform.node()}` with rank {mprank}')

print('World:', comm.Get_size(w))
