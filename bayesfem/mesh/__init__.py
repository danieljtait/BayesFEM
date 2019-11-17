"""
=====================================
Mesh Submodule (:mod:`bayesfem.mesh`)
=====================================

This module handles the construction of the mesh for the finite element method.

.. currentmodule:: bayesfem.mesh

Class overview
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Mesh
    TriangularMesh

"""
from bayesfem.mesh.mesh import Mesh
from bayesfem.mesh.triangularmesh import TriangularMesh
from bayesfem.mesh.intervalmesh import IntervalMesh
