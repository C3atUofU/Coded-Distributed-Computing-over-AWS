from mpi4py import MPI
from fractions import gcd
import numpy as np
import time
import copy
import pickle
import os
import itertools

# based on cdc_test_v19.py

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

r = 5  # computation load

num_workers = global_comm.Get_size() - 1  # Number of Workers (Nodes) | Excludes Master

m = num_workers/r

#numBatchesToSave = 10
dataDir = "data1"
#dataFile = "data_set"
#dataPath = dataDir + "/" + dataFile
num_files = 1000
total_data = 6 * 10 ** 8  # Number of key-value pairs to be sorted
key_max = 2 ** 16
value_max = 2 ** 16
vals_num_cols = 9
dataType = 'uint16'
array_len = int(total_data / num_files)


if rank == 0:  # Debugger
    print r

# matrix with dimensions (num_groups by r) that contains the ranks of the nodes of each shuffle group
worker_set = set([i for i in range(1,num_workers+1)])
node_index_start = np.linspace(1,num_workers+1,r,dtype=int,endpoint=False)

plac_group = [np.array(i,dtype=int) for i in list(itertools.product(range(m), repeat=r-1))]
num_plac_groups = len(plac_group)  # Number of Placement Groups

shuf_group_temp = [np.array(i,dtype=int) for i in list(itertools.product(range(m), repeat=r))]

for i in range(num_plac_groups):
    grp = np.append(plac_group[i],np.remainder(np.sum(plac_group[i]),m)) + node_index_start
    plac_group[i] = frozenset(grp)
#    if rank == 0:
#        print grp
#if rank == 0:
#    print '============'
#    time.sleep(10)
shuf_group = []
for i in range(len(shuf_group_temp)):
    grp = frozenset(shuf_group_temp[i] + node_index_start)
    if grp not in plac_group:
        shuf_group.append(grp)
#        if rank == 0:
#            print list(grp)


# Defines communication routing for shuffle groups with and without the master
shuffle_comms = {}

# Create comms (each node must do this)
for grp in shuf_group:
    group = list(grp)
    group.sort()
    newgroup = global_group.Incl(group)

    # Creates shuffle group's comm object
    shuffle_comms[grp] = comm0.Create(newgroup)  # Creates communication on group without master

# Bin (function) Assignment:
num_bins = num_workers
bin_edges = list(np.linspace(0,key_max,num_bins+1,dtype=int))

data_set = {}
file_order = []
data_partitions = [int(i) for i in total_data*np.linspace(0,1,num_plac_groups+1)]
if rank == 0:
    np.random.seed(1)
    file_order = list(np.random.permutation(num_files))
    
file_order = global_comm.bcast(file_order,root=0)

global_comm.Barrier()
if rank == 0:
    print 'Master: COMM SET UP COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]
    
# Worker nodes load files
num_temp_files = min(int(1+np.ceil(float(num_files)/float(num_plac_groups))),num_files)
temp_loaded_keys   = [[] for i in range(num_temp_files)]
temp_loaded_values = [[] for i in range(num_temp_files)]
temp_ld_fl_index   = [-1 for i in range(num_temp_files)]

temp_file_size = total_data/num_plac_groups+100
map_key_file = np.zeros((temp_file_size,),dtype = dataType)
map_value_file = np.zeros((temp_file_size,vals_num_cols),dtype = dataType)


grp_count = 0
local_bin_size = 0

for grp in plac_group:
    if rank in grp:
        first_index = int(np.floor(float(data_partitions[grp_count])/float(array_len)))
        last_index = min(first_index + num_temp_files - 1, num_files - 1)
        amt_data = 0
        amt_data_temp = 0
        for i in range(first_index,last_index+1):
            
            start_data = max(int(data_partitions[grp_count]  - i*array_len),0)
            stop_data  = max(0,min(array_len,int(data_partitions[grp_count+1]  - i*array_len)))
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

        data_set[grp] = {}
        indeces_bins = np.digitize(map_key_file[0:amt_data],bin_edges[1:num_bins+1])
        data_set[grp]['bin_size'] = np.array([np.sum(indeces_bins==i) for i in range(num_bins)])
        data_set[grp]['binned_keys'] = {}
        data_set[grp]['binned_values'] = {}
        for j in range(num_bins):
            data_set[grp]['binned_keys'][j+1] = map_key_file[np.where(indeces_bins==j)]
            data_set[grp]['binned_values'][j+1] = map_value_file[np.where(indeces_bins==j)]
            if j == rank - 1:
                local_bin_size += data_set[grp]['bin_size'][j]
    else:
        data_set[grp] = {}
    grp_count += 1
if rank != 0:
    del map_key_file, map_value_file, temp_loaded_keys, temp_loaded_values, data_partitions
                

keys_sort_local = np.zeros((local_bin_size,), dtype=dataType)
local_index = 0
values_local = np.zeros((local_bin_size, vals_num_cols), dtype=dataType)

    # Cycle through files:
for grp in plac_group:
    if rank in grp:
        bin_size = data_set[grp]['bin_size'][rank - 1]
        keys_sort_local[local_index:local_index + bin_size] = \
            data_set[grp]['binned_keys'][rank]
        del data_set[grp]['binned_keys'][rank]
        values_local[local_index:local_index + bin_size, :] = \
            data_set[grp]['binned_values'][rank]
        del data_set[grp]['binned_values'][rank]
        local_index += bin_size

global_comm.Barrier()
if rank == 0:
    print 'Master: MAP FUNCTIONS COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]


# ENCODE: Define packets to be transmitted (or used for decoding):

# Determine node requests of multicasting groups    
# Cycle through multicast groups
req_files = {}
for grp in shuf_group:
    req_files[grp] = {}
    if rank in grp:
        grp_list = list(grp)
        grp_list.sort()
        grp_array = np.array(grp_list)
        for i in range(r):
            recv_node = int(grp_array[i])
            if rank != recv_node:
                file = grp_array - node_index_start
                if i < r-1:
                    file[i] = np.remainder(file[r-1]-np.sum(file[0:r-1])+file[i],m)
                else:
                    file[r-1] = np.remainder(np.sum(file[0:r-1]),m)
                file += node_index_start
                req_files[grp][recv_node] = frozenset(file)


# Determine Multicasting Group Order, important for memory. It's also fast
grps_to_encode_ordered = {}
local_files = [grp for grp in plac_group if rank in grp]
grps_unassigned = [grp for grp in shuf_group if rank in grp]
for file in local_files:
    assigned_grps = []

    for grp in grps_unassigned:
        if len(grp - file) == 1:
            assigned_grps.append(grp)
    for grp in assigned_grps:
        grps_unassigned.remove(grp)
    grps_to_encode_ordered[file] = assigned_grps

transmit_data = {}
for file in local_files:
        
    for grp in grps_to_encode_ordered[file]:

        transmit_data[grp] = {}

        for xt_node in grp:
            transmit_data[grp][xt_node] = {'max_word_len': 0}

        for recv_node in grp:
            if rank != recv_node:
                tx_keys = None
                tx_values = None

                k = req_files[grp][recv_node]

                msg_size = data_set[k]['bin_size'][recv_node - 1]

                transmit_data[grp][recv_node]['req_keys'] = data_set[k]['binned_keys'][recv_node]
                transmit_data[grp][recv_node]['req_values'] = data_set[k]['binned_values'][recv_node]
                indexing = list(np.round(np.linspace(0, msg_size, r)))
                indexing = [int(x) for x in indexing]
                c = 0
                for xt_node in sorted(grp):
                    if xt_node != recv_node:
                        transmit_data[grp][xt_node][recv_node] = {'tx_index': indexing[c:c + 2],
                                                                  'tx_word_len': indexing[c + 1] - indexing[c]}

                        if  transmit_data[grp][xt_node]['max_word_len'] < indexing[c + 1] - indexing[c]:
                            transmit_data[grp][xt_node]['max_word_len'] = indexing[c + 1] - indexing[c]
                        c += 1

        for xt_node in grp:
            pad_val = int(bin_edges[xt_node - 1])
            max_word_length = transmit_data[grp][xt_node]['max_word_len']
            transmit_data[grp][xt_node]['keys_codeword'] = np.zeros((max_word_length,), dtype=dataType)
            transmit_data[grp][xt_node]['values_codeword'] = np.zeros((max_word_length, vals_num_cols), dtype=dataType)
            for recv_node in grp:
                if xt_node != recv_node and rank != recv_node:
                    msg_length = transmit_data[grp][xt_node][recv_node]['tx_word_len']
                    indexing = transmit_data[grp][xt_node][recv_node]['tx_index']
                    transmit_data[grp][xt_node]['keys_codeword'][0:msg_length] ^= \
                        transmit_data[grp][recv_node]['req_keys'][indexing[0]:indexing[1]]

                    if msg_length < max_word_length:
                        transmit_data[grp][xt_node]['keys_codeword'][msg_length:max_word_length] ^= pad_val
                    
                        
                    transmit_data[grp][xt_node]['values_codeword'][0:msg_length, :] ^= \
                        transmit_data[grp][recv_node]['req_values'][indexing[0]:indexing[1], :]
    del data_set[file]

del data_set

global_comm.Barrier()
if rank == 0:
    print 'Master: DATA ENCODE COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]

# Shuffle data:
xt_data = []
rcvd_keys = {}
rcvd_values = {}
rcv_len_total = 0
for grp in shuf_group:
    if rank in grp:
        rcvd_keys[grp] = {}
        rcvd_values[grp] = {}
if rank == 0:
    t1 = time.time()
for xt_node in range(1,num_workers+1):
    global_comm.Barrier()
    if rank == 0:
        t2 = time.time()
        print t2 - t1
        print xt_node
        t1 = t2
        
        
    for grp in shuf_group:
        if (rank in grp) and (xt_node in grp):
            sorted_group = sorted(grp)
            xt_node_grp_rank = sorted_group.index(xt_node)
            if xt_node == rank:
                xt_data = transmit_data[grp][xt_node]['keys_codeword']
            xt_data = shuffle_comms[grp].bcast(xt_data, root=xt_node_grp_rank)
            if xt_node != rank:
                rcvd_keys[grp][xt_node] = xt_data
                rcv_len_total += len(xt_data)
            if xt_node == rank:
                xt_data = transmit_data[grp][xt_node]['values_codeword']
            xt_data = shuffle_comms[grp].bcast(xt_data, root=xt_node_grp_rank)
            if xt_node != rank:
                rcvd_values[grp][xt_node] = xt_data

global_comm.Barrier()
if rank == 0:
    print 'Master: DATA SHUFFLE COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]

for grp in shuf_group:
    if rank in grp:
        for xt_node in grp:
            if xt_node != rank:
                tx_len = len(rcvd_keys[grp][xt_node])
                if np.remainder(r - 2, 2) == 1:
                    pad_val = bin_edges[xt_node - 1]
                else:
                    pad_val = 0
                pad_length = tx_len - transmit_data[grp][xt_node]['max_word_len']
                rcvd_keys[grp][xt_node] ^= np.pad(transmit_data[grp][xt_node]['keys_codeword'],
                                                  (0, pad_length), 'constant', constant_values=(0, pad_val))
                
                del transmit_data[grp][xt_node]['keys_codeword']
                
                pad_val = bin_edges[xt_node - 1]
                dc_data_len = np.argmax(rcvd_keys[grp][xt_node] == pad_val)
                if not dc_data_len:
                    dc_data_len = tx_len

                rcvd_keys[grp][xt_node] = rcvd_keys[grp][xt_node][0:dc_data_len]

                rcvd_values[grp][xt_node] ^= np.pad(transmit_data[grp][xt_node]['values_codeword'],
                                                    ((0, pad_length), (0, 0)), mode='constant', constant_values=(0, 0))
                
                del transmit_data[grp][xt_node]['values_codeword']
                
                rcvd_values[grp][xt_node] = rcvd_values[grp][xt_node][0:dc_data_len]

del transmit_data

global_comm.Barrier()
if rank == 0:
    print 'Master: DATA DECODE COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]
#    print 'Total time:',timeStamps[timeCount]-timeStamps[0]


keys_sort_rcvd = np.zeros((rcv_len_total+local_index,), dtype=dataType)
values_sort_rcvd = np.zeros((rcv_len_total+local_index, vals_num_cols), dtype=dataType)
keys_sort_rcvd[0:local_index] = keys_sort_local

del keys_sort_local

values_sort_rcvd[0:local_index,:] = values_local

del values_local

placement_index = local_index
for grp in shuf_group:
    if rank in grp:
        for xt_node in grp:
            if xt_node != rank:
                dc_data_len = len(rcvd_keys[grp][xt_node])
                keys_sort_rcvd[placement_index:placement_index + dc_data_len] = rcvd_keys[grp][xt_node]
                del rcvd_keys[grp][xt_node]
                values_sort_rcvd[placement_index:placement_index + dc_data_len, :] = rcvd_values[grp][xt_node]
                del rcvd_values[grp][xt_node]
                placement_index += dc_data_len

del rcvd_values, rcvd_keys

#keys_sort_local = np.append(keys_sort_rcvd[0:placement_index], keys_sort_local)

#del keys_sort_rcvd

indeces_sort = np.argsort(keys_sort_rcvd[0:placement_index])

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

