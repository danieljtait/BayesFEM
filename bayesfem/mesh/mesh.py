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
    def n_elements(self):
        """ number of elements in the mesh. """
        return self.elements.shape[0]