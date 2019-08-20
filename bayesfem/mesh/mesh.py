class Mesh:
    """ Base mesh class. """
    def __init__(self, points):
        self._points = points

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

    @property
    def element_type(self):
        return self._element_type

    @property
    def npoints(self):
        return self._points.shape[0]
