"""
with open('output_one_picture/dcdl_conv_1_out2_raw.txt', 'r') as infile, open(
        'output_one_picture/dcdl_conv_1_out2_boiled.txt', 'w') as outfile:
    temp = infile.read().replace("[", "").replace("]","")
    outfile.write(temp)"""


import numpy as np
import matplotlib.pyplot as plt

