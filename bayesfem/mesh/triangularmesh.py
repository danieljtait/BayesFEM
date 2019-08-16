import numpy as np
from scipy.spatial import Delaunay

from .mesh import Mesh

__all__ = ['TriangularMesh', ]


class TriangularMesh(Delaunay, Mesh):
    """ Triangular mesh on R2. """
    def __init__(self, points):
        """ Creates a TriangularMesh instance. """
        super(TriangularMesh, self).__init__(points)

        self._elements = self.simplices
        self._boundary_nodes = np.unique(self.convex_hull)

        self._element_type = 'Triangular'
