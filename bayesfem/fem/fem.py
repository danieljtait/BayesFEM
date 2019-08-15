

class BaseFEM:
    """ Base FEM object. """
    def __init__(self, mesh):
        """ Instatiates a BaseFEM object. """
        self._mesh = mesh

    @property
    def mesh(self):
        """ Underlying mesh on which the FEM is being solved. """
        return self._mesh
