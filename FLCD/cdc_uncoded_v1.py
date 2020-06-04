from mpi4py import MPI
from fractions import gcd
import numpy as np
import time
import copy
import pickle
import os

global_comm = MPI.COMM_WORLD
rank = global_comm.rank

global_comm.Barrier()
if rank == 0:
    timeCount = 0
    timeStamps = np.zeros((20,))
    timeStamps[timeCount] = time.time()
    print 'Master: COMM SET UP START'
# Set up for comm creation for shuffle groups
global_group = global_comm.Get_group()
comm0 = global_comm.Create(global_group)

num_workers = global_comm.Get_size() - 1
dataDir = "data1"
num_files = 1000
total_data = 6 * 10 ** 8  # Number of key-value pairs to be sorted
key_max = 2 ** 16
value_max = 2 ** 16
vals_num_cols = 9
dataType = 'uint16'
array_len = int(total_data / num_files)



# Create comms (each node must do this)

group = [i for i in range(1,num_workers+1)]
newgroup = global_group.Incl(group)
shuffle_comms = comm0.Create(newgroup)  # Creates communication on group without master

# Bin (function) Assignment:
num_bins = num_workers
bin_edges = list(np.linspace(0,key_max,num_bins+1,dtype=int))


data_set = {}
file_order = []
data_partitions = [int(i) for i in total_data * np.linspace(0,1,num_workers+1)]
if rank == 0:
    np.random.seed(1)
    file_order = list(np.random.permutation(num_files))
#    print data_partitions
    
file_order = global_comm.bcast(file_order,root=0)

global_comm.Barrier()
if rank == 0:
    print 'Master: COMM SET UP COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]
    
# Worker nodes load files
num_temp_files = min(int(1+np.ceil(float(num_files)/float(num_workers))),num_files)
temp_loaded_keys   = [[] for i in range(num_temp_files)]
temp_loaded_values = [[] for i in range(num_temp_files)]
temp_ld_fl_index   = [-1 for i in range(num_temp_files)]

temp_file_size = total_data/num_workers+100
map_key_file = np.zeros((temp_file_size,),dtype = dataType)
map_value_file = np.zeros((temp_file_size,vals_num_cols),dtype = dataType)


local_bin_size = 0


if rank != 0:
    first_index = int(np.floor(float(data_partitions[rank-1])/float(array_len)))
    last_index = min(first_index + num_temp_files - 1, num_files - 1)
    amt_data = 0
    amt_data_temp = 0
    for i in range(first_index,last_index+1):
        
        start_data = max(int(data_partitions[rank-1]  - i*array_len),0)
        stop_data  = max(0,min(array_len,int(data_partitions[rank]  - i*array_len)))
        amt_to_append = stop_data - start_data
        amt_data_temp += amt_to_append
    
        if amt_to_append > 0:
            if i in temp_ld_fl_index:
                temp_index = temp_ld_fl_index.index(i)
            else:
                key_fname = dataDir + "/keys_%03d.npy" % file_order[i]
                value_fname = dataDir + "/values_%03d.npy" % file_order[i]
                temp_index = np.argmin(temp_ld_fl_index)
                temp_ld_fl_index[temp_index] = i
                temp_loaded_keys[temp_index] = np.load(key_fname)
                temp_loaded_values[temp_index] = np.load(value_fname)
    
            temp_loaded_keys[temp_index][start_data:stop_data]
            map_key_file[amt_data:amt_data_temp] = temp_loaded_keys[temp_index][start_data:stop_data]
            map_value_file[amt_data:amt_data_temp,:] = temp_loaded_values[temp_index][start_data:stop_data,:]
            amt_data += amt_to_append
    
    indeces_bins = np.digitize(map_key_file[0:amt_data],bin_edges[1:num_bins+1])
    data_set['bin_size'] = np.array([np.sum(indeces_bins==i) for i in range(num_bins)])
    data_set['binned_keys'] = []
    data_set['binned_values'] = []
    for j in range(num_bins):
        data_set['binned_keys'].append(map_key_file[np.where(indeces_bins==j)])
        data_set['binned_values'].append(map_value_file[np.where(indeces_bins==j),:])
                          
        if j == rank - 1:
            local_bin_size += data_set['bin_size'][j]

    del map_key_file, map_value_file, temp_loaded_keys, temp_loaded_values, data_partitions
                



keys_sort_local = np.zeros((local_bin_size,), dtype=dataType)
local_index = 0
values_local = np.zeros((local_bin_size, vals_num_cols), dtype=dataType)


global_comm.Barrier()
if rank == 0:
    print 'Master: MAP FUNCTIONS COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]


# Shuffle data:
rcvd_keys = {}
rcvd_values = {}
rcv_len_total = 0
for xt_node in range(1,num_workers+1):
    if rank != 0:
        rcvd_keys[xt_node] = shuffle_comms.scatter(data_set['binned_keys'],root=xt_node-1)
        rcvd_values[xt_node] = shuffle_comms.scatter(data_set['binned_values'],root=xt_node-1)
        rcv_len_total += len(rcvd_keys[xt_node])
    else:
        print xt_node
    global_comm.Barrier()

del data_set

global_comm.Barrier()
if rank == 0:
    print 'Master: DATA SHUFFLE COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]

# REDUCE

keys_sort_rcvd = np.zeros((rcv_len_total,), dtype=dataType)
values_sort_rcvd = np.zeros((rcv_len_total, vals_num_cols), dtype=dataType)

placement_index = 0
if rank != 0:
    for xt_node in range(1,num_workers+1):
        data_len = len(rcvd_keys[xt_node])
        if rank == 1:
            print data_len
        keys_sort_rcvd[placement_index:placement_index + data_len] = rcvd_keys[xt_node]
        del rcvd_keys[xt_node]
        values_sort_rcvd[placement_index:placement_index + data_len, :] = rcvd_values[xt_node]
        del rcvd_values[xt_node]
        placement_index += data_len

del rcvd_values, rcvd_keys

#keys_sort_local = np.append(keys_sort_rcvd[0:placement_index], keys_sort_local)

#del keys_sort_rcvd

indeces_sort = np.argsort(keys_sort_rcvd)

sorted_keys = keys_sort_rcvd[indeces_sort]

del keys_sort_rcvd

#values_sort_rcvd = values_sort_rcvd[indeces_sort,:]

num_val_matrices = 100

num_keyval_pairs = len(indeces_sort)

matrix_partitions = list(np.linspace(0,num_keyval_pairs,num_val_matrices+1,dtype='int'))

val_matrices = []
for i in range(num_val_matrices):
    val_matrices.append(values_sort_rcvd[indeces_sort[matrix_partitions[i]:matrix_partitions[i+1]],:])

del values_sort_rcvd, indeces_sort

global_comm.Barrier()
if rank == 0:
    print 'Master: REDUCE COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]
    print '\t Total time:', timeStamps[timeCount] - timeStamps[0]
else:
    time.sleep(0.5*rank)
    print '==============='
    print rank
    print bin_edges[rank-1],bin_edges[rank]
    print num_keyval_pairs
    print sorted_keys

