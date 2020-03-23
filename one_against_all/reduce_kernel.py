import helper_methods as help
import numpy as np

if __name__ == '__main__':

    kernel_approximation = np.load('data/kernel_approximation.npy')
    for i, kernel in enumerate(kernel_approximation):
        reduced_kernel = help.reduce_kernel(kernel, mode='sum')
        help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)),  28,
                                     'Sum of all found SLS Formel for kernel {}'.format(i),
                                     set_vmin_vmax= False)

        reduced_kernel = help.reduce_kernel(kernel, mode='mean')
        help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'mean of all found SLS Formel for kernel {}'.format(i),
                                     set_vmin_vmax= False)

        reduced_kernel = help.reduce_kernel(kernel, mode='min_max')
        help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'min_max of all found SLS Formel for kernel {}'.format(i))

        reduced_kernel = help.reduce_kernel(kernel, mode='norm')
        help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'norm of all found SLS Formel for kernel {}'.format(i))

