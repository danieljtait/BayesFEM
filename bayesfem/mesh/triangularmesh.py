import numpy as np
from scipy.spatial import Delaunay
from .mesh import Mesh
import tensorflow as tf

# Import additional utilities for generating
# a mesh from matlab
try:
    import matlab.engine
except ImportError:
    pass

__all__ = ['TriangularMesh', ]


class TriangularMesh(Mesh):

    _element_type = 'Triangular'

    def __init__(self,
                 points,
                 elements,
                 boundary_node_indices,
                 name='TriangularMesh'):

        with tf.name_scope('Init'):
            self._points = points
            self._elements = elements

            # ToDo: replace _boundary_nodes in older classes with
            #   the new boundary_node_indices
            self._boundary_node_indices = boundary_node_indices
            self._boundary_nodes = boundary_node_indices

            # get the indices of the interior nodes
            interior_node_indices = np.array([i for i in range(self.npoints)
                                              if not i in self.boundary_node_indices],
                                             dtype=np.intp)
            self._interior_node_indices = interior_node_indices

            X, Y = self.points[self.elements].T
            self._element_areas = .5 * ((X[0] - X[2]) * (Y[1] - Y[0])
                                        - (X[0] - X[1]) * (Y[2] - Y[0]))

            cx = np.mean(X, axis=0)
            cy = np.mean(Y, axis=0)

            self._barycenters = np.column_stack((cx, cy))

    @staticmethod
    def from_verts_by_matlab(verts, hmax):
        """ Creates a mesh of polygon defined by verts using the Matlab Engine.

        Parameters
        ----------
        verts : 2d list
            list of vertices [[x1, y1], ..., [xn, yn]]

        hmax : float
            Maximum edge size for generating the mesh
        """
        eng = matlab.engine.start_matlab()
        p, e, t = (np.asarray(item)
                   for item in eng.meshfrompolyverts(matlab.double(verts),
                                                     hmax,
                                                     nargout=3))
        eng.quit()

        # additional postprocessing on d
        # convert from matlab indexing to
        # python indexing
        t[:3, :] -= 1
        e[:2, :] -= 1

        t = np.asarray(t, dtype=np.intp)
        e = np.asarray(e, dtype=np.intp)

        boundary_node_indices = np.unique(e[:2, :])

        return TriangularMesh(p.T, t[:3, :].T, boundary_node_indices)

    @property
    def barycenters(self):
        return self._barycenters

    @property
    def element_areas(self):
        return self._element_areas



class TriangularMeshOld(Delaunay):
    """ Triangular mesh on R2.

    ToDo:
        Should have mesh as a parent... problem is that delauney already
        has n_points as an attribute
    """
    def __init__(self, points):
        """ Creates a TriangularMesh instance. """
        super(TriangularMesh, self).__init__(points)

        self._boundary_nodes = np.unique(self.convex_hull)
        self._element_type = 'Triangular'


    @property
    def elements(self):
        return self.simplices

    @property
    def boundary_nodes(self):
        """ Indices of boundary points. """
        return self._boundary_nodes

    @property
    def boundary_node_indices(self):
        return self._boundary_nodes

    @property
    def n_elements(self):
        """ number of elements in the mesh. """
        return self.elements.shape[0]

    @property
    def element_type(self):
        return self._element_type

    #@property
    #def npoints(self):
    #    return self._points.shape[0]

    def interior_node_indices(self, dtype=np.int32):
        """ Returns the indices for those nodes not on the boundary.

        Parameters
        ----------
        dtype: np.dtype
            Data type of integer array

        Returns
        -------
        indices : array, dtype = dtype
            Indices of `self.points` for nodes not on the boundary.
        """
        return np.array([i for i in range(self.npoints) if not i in self.boundary_nodes], dtype=np.int32)

    def get_quadrature_nodes(self):
        x, y = self.points[self.elements].T
        return np.array([np.sum(x, axis=0)/3.,
                         np.sum(y, axis=0)/3.])