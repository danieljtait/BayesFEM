import numpy as np
import tensorflow as tf
from .mesh import Mesh

__all__ = ['IntervalMesh', ]


class IntervalMesh(Mesh):
    """ Interval mesh on R """
    def __init__(self, points):
        """ Creates an IntervalMesh instance. """
        self._points = np.sort(points)

        elements = np.column_stack([np.arange(0, self.npoints-1),
                                    np.arange(1, self.npoints)])

        self._elements = elements
        self._boundary_nodes = [0, self.npoints-1]
        self._nboundary_nodes = 2

        self._element_type = 'Interval'

    def get_quadrature_nodes(self):
        xa, xb = self.points[self.elements].T
        return .5*(xb + xa)

    def linear_interpolation_operator(self, index_points):
        """ Returns the linear operator that carries out interpolation of the solution at node points.

        Parameters
        ----------
        index_points : array, shape=[..., nobs, ]
            points at which we want to interpolate

        Returns
        -------
        O : batched array, shape = [..., nobs, mesh.npoints]
            linear operator such that O @ u interpolates u[mesh.points] onto
            index_points


        .. note::

            The matrix of the operator returned will have 2 entries per row, and so
            will be sparse when `mesh.npoints >> index_points.shape[-1]`.

        """
        # linear interpolation in R
        lower_bounds = ((index_points[:, None] > self.points)
                        * np.arange(self.npoints)[None, :])

        # lower node index of the elements
        elements = np.max(lower_bounds, axis=-1)
        # take advantage of points being sorted
        elements = np.column_stack((elements, elements + 1))

        # get the lower and upper nodes
        ta, tb = self.points[elements].T

        # construct the basis functions
        # again uses the fact that mesh.points is sorted
        # ToDo : make `element_volume` a method of mesh
        element_lengths = np.diff(self.points)

        phi1 = (tb - index_points) / element_lengths[elements[:, 0]]
        phi2 = (index_points - ta) / element_lengths[elements[:, 0]]

        # create the linear interpolation operator
        nobs = index_points.shape[0]
        Oa = np.zeros((nobs, self.npoints), dtype=self.points.dtype)
        Oa[np.arange(nobs), elements[:, 0]] = phi1

        Ob = np.zeros((nobs, self.npoints), dtype=self.points.dtype)
        Ob[np.arange(nobs), elements[:, 1]] = phi2

        return tf.constant(Oa + Ob)
