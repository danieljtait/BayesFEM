"""
Triangular mesh in R^2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bayesfem.mesh import Mesh
import tensorflow.compat.v2 as tf
import numpy as np

__all__ = ['TriangularMesh', ]

# matlab function to triangulate a region.
meshfrompolyverts = (
    'function [p, e, t] = meshfrompolyverts(X, hmax)\n'
    '  pg = polyshape(X(:, 1), X(:, 2))\n',
    '  tr = triangulation(pg);\n',
    '  model = createpde();\n',
    '  geom = geometryFromMesh(model, tr.Points\', tr.ConnectivityList\');\n',
    '  femmesh = generateMesh(model, \'Hmax\', hmax, \'GeometricOrder\', \'linear\');\n',
    '  [p, e, t] = meshToPet(femmesh);\n',
)

class TriangularMesh(Mesh):
    """ Triangular mesh in R2 """

    def __init__(self,
                 nodes,
                 elements,
                 boundary_node_indices,
                 name='TriangularMesh'):

        super(TriangularMesh, self).__init__(nodes)

        self._elements = elements
        self._element_type = 'Triangular'

        self._boundary_node_indices = boundary_node_indices
        self._interior_node_indices = [i for i in range(self.n_nodes)
                                       if not i in self.boundary_node_indices]

        # get the triangle areas
        X, Y = [tf.reshape(
            tf.gather(self.nodes[:, i],
                      [*self.elements.flat]),
            self.elements.shape) for i in range(2)]
        self._element_volumes = .5 * ((X[..., 0] - X[..., 2]) * (Y[..., 1] - Y[..., 0])
                                       - (X[..., 0] - X[..., 1]) * (Y[..., 2] - Y[..., 0]))
        # and the barycenters
        cx = tf.reduce_mean(X, axis=-1)
        cy = tf.reduce_mean(Y, axis=-1)
        self._barycenters = tf.stack((cx, cy))

        self._element_dim = 3

    @staticmethod
    def from_verts_by_matlab(verts, hmax):
        """ Creates a mesh of the polygon defined by verts using the matlab engine

        Parameters
        ----------
        verts : 2d list
            list of vertices [[x1, y1], ..., [xn, yn]]

        hmax : float
            Maximum (approx.) edge size for generating the mesh

        Returns
        -------
        trimesh : TriangularMesh
            Returns the triangular mesh created from the polygon.
        """
        try:
            import os
            import matlab.engine
        except ImportError:
            raise ImportError

        # start the matlab engine
        eng = matlab.engine.start_matlab()

        # create the file defining the matlab function to be used
        fname = 'meshfrompolyverts.m'
        path = os.path.join(os.getcwd(), fname)

        with open(path, 'w') as f:
            f.write(''.join(meshfrompolyverts))

        p, e, t = (np.asarray(item)
                   for item in eng.meshfrompolyverts(matlab.double(verts),
                                                     hmax,
                                                     nargout=3))
        eng.quit()

        # clean up the tmp file
        try:
            os.remove(path)
        except OSError as e:
            print("Error: {} -- {}.".format(e.filename, e.strerror))

        # additional postprocessing on d
        # convert from matlab indexing to
        # python indexing
        t[:3, :] -= 1
        e[:2, :] -= 1


        t = np.asarray(t, dtype=np.intp)
        e = np.asarray(e, dtype=np.intp)

        boundary_node_indices = np.unique(e[:2, :])

        trimesh = TriangularMesh(p.T, t[:3, :].T, boundary_node_indices)
        trimesh._e = e
        return trimesh
