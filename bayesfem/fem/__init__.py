"""
==================
`mod`:bayesfem.fem
==================
"""
from .fem import BaseFEM
from .poisson import Poisson
from .linearsecondorderelliptic import LinearSecondOrderElliptic
from .femassembler import FEMAssembler
from . import boundary_util
