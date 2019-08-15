

class BaseFEM:
    """ Base FEM object. """
    def __init__(self, mesh, dtype=tf.float32):
        """ Instatiates a BaseFEM object. """
        self._mesh = mesh
        self._dtype = dtype

    @property
    def mesh(self):
        """ Underlying mesh on which the FEM is being solved. """
        return self._mesh

    @property
    def dtype(self):
        """ Common data dtype for the tensors comprising the solver. """
        return self._dtype
