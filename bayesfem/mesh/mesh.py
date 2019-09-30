import numpy as np


class Mesh:
    """ Base mesh class. """

    @property
    def points(self):
        """ Points defining the mesh. """
        return self._points

    @property
    def elements(self):
        """ Indices of points specifying each element. """
        return self._elements

    @property
    def boundary_nodes(self):
        """ Indices of boundary points. """
        return self._boundary_nodes

    @property
    def boundary_node_indices(self):
        return self._boundary_node_indices

    @property
    def n_elements(self):
        """ number of elements in the mesh. """
        return self.elements.shape[0]

    @property
    def element_type(self):
        return self._element_type

    @property
    def npoints(self):
        return self._points.shape[0]

    @property
    def nboundary_nodes(self):
        return self._nboundary_nodes

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