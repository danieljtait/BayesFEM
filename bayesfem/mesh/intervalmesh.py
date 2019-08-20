import numpy as np

from .mesh import Mesh

__all__ = ['IntervalMesh', ]


class IntervalMesh(Mesh):
    """ Interval mesh on R """
    def __init__(self, points):
        """ Creates an IntervalMesh instance. """
        points = np.sort(points)
        super(IntervalMesh, self).__init__(points)

        elements = np.column_stack([np.arange(0, self.npoints-1),
                                    np.arange(1, self.npoints)])

        self._elements = elements
        self._boundary_nodes = [0, self.npoints-1]

        self._element_type = 'Interval'
