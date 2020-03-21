import numpy as np

def one_class_against_all(array_label, one_class=-1):
    """
converts an array with one_hot_vector for any number of classes into a one_hot_vector,
 whether an example belongs to one class or not
    """
    label_one_class_against_all = np.zeros(array_label.shape, dtype=int)
    for i, one_hot_vector in enumerate(array_label):
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all[i,0]=1
        else :
            label_one_class_against_all[i,-1]=1
    return label_one_class_against_all



"""
    label_one_class_against_all = []
    shape_label = array_label.shape
    length_one_hot_vector = shape_label[-1]
    for one_hot_vector in array_label:
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all.append([1,0,0,0,0,0,0,0,0,0])
        else :
            label_one_class_against_all.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    return np.array(label_one_class_against_all).reshape(shape_label)
    
    """