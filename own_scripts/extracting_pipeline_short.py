import numpy as np
import parallel_sls.python_wrapper.sls_wrapper as sls_wrapper
import parallel_sls.python_wrapper.data_wrapper as data_wrapper
from sklearn import tree
import math

# stacked bilder aufeinander und zwar jededen zweiten pixel
def window_stack(arr, stepsize=2, width=4):
    dims = arr.shape
    out_arr = np.stack([arr[picture, row:row+width, col:col+width, :].flatten()
                        for picture in range(0, dims[0])  for row in range(0, dims[1] - width + 1, stepsize) for col in range(0, dims[2] - width + 1, stepsize) ], axis=0)
    print(out_arr.shape)
    return out_arr


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIG +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
out_arr = []
num_blocks, start_block = 1,1
pooling_at = [1, 2, 3]
start_res_in = 32
start_num_ex_in = 10 * 35
k = 500

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PIPELINE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start_line = 0
start_res = start_res_in
start_num_ex = start_num_ex_in

for i in range(start_block - 1, num_blocks):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Get training and label set
    lines_to_take = start_res * start_res * start_num_ex
    training_set = np.reshape(np.loadtxt('datasets.txt', skiprows=start_line, max_rows=lines_to_take),
                              (start_num_ex, start_res, start_res, -1))
    start_line += lines_to_take

    # Need padding for convolution
    npad = ((0, 0), (1, 1), (1, 1), (0, 0))
    training_set_padded = np.pad(training_set, pad_width=npad, mode='constant', constant_values=0)

    if (i + 1) in pooling_at:
        start_res = int(math.ceil(start_res/2))

    lines_to_take = start_res * start_res * start_num_ex
    label_set = np.reshape(np.loadtxt('datasets.txt', skiprows=start_line, max_rows=lines_to_take),
                           (start_num_ex, start_res, start_res, -1))
    start_line += lines_to_take

    if (i + 1) in pooling_at:
        start_num_ex *= 4

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Reshape both sets to be fitting
    training_set_stack = window_stack(  training_set_padded, stepsize=2, width=4)
    flat_training_set = np.maximum(training_set_stack, 0).astype(np.bool) # 2,4 for pooled layer
    label_set = np.maximum(label_set, 0).astype(np.bool)

    #unique_instances, unique_indices = np.unique(flat_training_set,axis=0, return_index=True)
    #unique_instances_labeled, unique_indices_labeled = np.unique(np.concatenate((flat_training_set, np.expand_dims(label_set[:,:,:,0].flatten(), axis = 1)), axis = 1),axis=0, return_index=True)
    #print("u", unique_instances.shape)
    #print("ul", unique_instances_labeled.shape)


    #flat_training_set = flat_training_set[unique_indices]
    random_indices = p = np.random.permutation(len(flat_training_set))
    flat_training_set = flat_training_set[p]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Actual extraction
    for task in range(min(label_set.shape[3], 5)):
        label_set_split = label_set[:,:,:,task].flatten()#[unique_indices]
        label_set_split = label_set_split[p]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Dataset stats
        num_vars = flat_training_set.shape[1] % 8 + flat_training_set.shape[1]
        vars_per_var = int(num_vars / 8)

        first_split = int(flat_training_set.shape[0] * 2/3)
        second_split = int(flat_training_set.shape[0] * 2/3) + int((flat_training_set.shape[0] - int(flat_training_set.shape[0] * 2/3)) * 1/2)

        print(  'first_split {} , size split {}, flat_training_set.shape[0] - second_split {}'.format(                     first_split,
                              second_split - first_split,
                              flat_training_set.shape[0] - second_split))


        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(flat_training_set[:first_split], label_set_split[:first_split])
        print('clf.score(flat_training_set[first_split:second_split], label_set_split[first_split:second_split: {}])'
              .format(clf.score(flat_training_set[first_split:second_split], label_set_split[first_split:second_split])))
        print('clf.score(flat_training_set[:first_split], label_set_split[:first_split] {}'.format(clf.score(flat_training_set[:first_split], label_set_split[:first_split])))
        print('Depth of tree {}'.format(clf.get_depth()))

