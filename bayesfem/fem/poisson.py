from .fem import BaseFEM


class Poisson(BaseFEM):

    def __init__(self, mesh, name='Poisson'):
        """ Instatiates a Poisson FEM solver"""
        super(Poisson, self).__init__(mesh)
