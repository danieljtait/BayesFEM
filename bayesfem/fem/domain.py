"""
A domain is the collection of a mesh and the boundary information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

__all__ = ['Domain', ]

class Domain:
    def __init__(self, mesh, boundary):
        self._mesh = mesh
        self._boundary = boundary

    @property
    def mesh(self):
        return self._mesh

    @property
    def boundary(self):
        return self._boundary
