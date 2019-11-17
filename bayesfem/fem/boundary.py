
__all__ = ['Boundary', 'DirichletBoundary']

class Boundary:
    """ Base boundary class """
    def __init__(self):
        pass

    @property
    def boundary_condition_type(self):
        return self._boundary_condition_type

class DirichletBoundary(Boundary):
    """
    Dirichlet boundary conditions u(x) = g(x) on the boundary.
    """
    def __init__(self, g):
        self._g = g
        self._boundary_condition_type = 'Dirichlet'

    @property
    def g(self):
        return self._g
