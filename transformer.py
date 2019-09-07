#!/usr/bin/env python

'''
Example of the blocksparse transformer on enwik8.

To download data:

wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip -d /tmp
'''

import argparse
import numpy       as np
import tensorflow  as tf
import blocksparse as bs
# from mpi4py import MPI
from attention import blocksparse_attention_impl
from attention import attention_impl


