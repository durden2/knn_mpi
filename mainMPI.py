from mpi4py import MPI
import numpy as np
import time

from knn import knn

class mpi:
    def __init__(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._proces_count = self._comm.Get_size()

    def run(self, clf, params_array):
        params_count = len(params_array)
        number_of_elements_per_proces = int(params_count / self._proces_count)

        sendbuf = None
        if self._rank == 0:
            #prepare data to send
            #reshape params array to fit the same amount of data to every process
            sendbuf = params.reshape([self._proces_count, number_of_elements_per_proces])

        #recieve buffer for one process
        recvbuf = np.empty(number_of_elements_per_proces, dtype='i')

        #start time measurement
        start = time.time()

        #send reshaped array to all processes
        self._comm.Scatter(sendbuf, recvbuf, root=0)

        #prepare an array with results to send after calculation to root process
        results_for_single_process = []

        #perform an algorithm for every param revieced
        #and add to results array
        for i in recvbuf:
            result = (clf(i))
            results_for_single_process.append(result)

        #prepare recieve buffer for all processes
        recvbuf = None
        if self._rank == 0:
            recvbuf = np.empty([self._proces_count, number_of_elements_per_proces], dtype=np.float)

        #send results back to root
        results_for_single_process = np.array(results_for_single_process)
        self._comm.Gather(results_for_single_process, recvbuf, root=0)

        #when all finished show runtime
        #all results are now available in recvbuf
        if self._rank == 0:
            end = time.time()
            print('finished in ', end - start)

my_knn = knn()
params = np.arange(1, 288, 2, dtype='i')
mpi().run(my_knn.train, params)


