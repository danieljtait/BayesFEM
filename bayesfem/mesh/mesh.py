"""
Base mesh class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow.compat.v2 as tf

__all__ = ['Mesh', ]


@six.add_metaclass(abc.ABCMeta)
class Mesh:
    """ Abstract base class for meshes.

    #### Background

    The finite element method partitions the domain into a mesh
    of elements.

    """
    def __init__(self, nodes, dtype=tf.float32):
        """ Creates a Mesh instance. """
        self._nodes = tf.convert_to_tensor(
            nodes,
            dtype=dtype,
            name='nodes')

    @property
    def nodes(self):
        """ The nodes defining the mesh.

        Returns
        -------
        nodes : Tensor, shape (n_nodes, n_dims)
            The nodes defining the mesh.
        """
        return self._nodes

    @property
    def edges(self):
        """ The edges of the mesh

        Returns
        -------
        e : Tensor, shape (n_edges, 7)
            The edges of the mesh.
        """
        return self._edges

    @property
    def n_edges(self):
        """ The number of edge segments in the mesh. """
        return self._n_edges

    @property
    def elements(self):
        """ Elements defining the mesh.

        Returns
        -------
        elements : Integer Tensor, shape (n_elements, n_dims)
            Indices of nodes for each element
        """
        return self._elements

    @property
    def dtype(self):
        """ Data type of the mesh nodes.

        Returns
        -------
        dtype : tf.dtype
            Data type of the mesh nodes
        """
        return self.nodes.dtype

    @property
    def element_type(self):
        return self._element_type

    @property
    def element_volumes(self):
        return self._element_volumes

    @property
    def barycenters(self):
        """ Barycenters of the elements.

        Returns
        -------
        barycenters : Tensor, shape = (domain_dim, b1, ..., bB, mesh.n_elements)
            Barycenters of each element.
        """
        return self._barycenters

    @property
    def n_elements(self):
        """ Number of elements in the mesh. """
        return self.elements.shape[0]

    @property
    def n_nodes(self):
        """ Number of nodes in the mesh. """
        return self.nodes.shape[-2]

    @property
    def boundary_node_indices(self):
        """ Indices of nodes on the mesh boundary """
        return self._boundary_node_indices

    @property
    def interior_node_indices(self):
        """ Indices of nodes in the interior """
        return self._interior_node_indices
