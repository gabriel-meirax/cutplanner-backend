"""
CutPlanner - Sistema de Otimização de Cortes para Serralherias

Um sistema inteligente que otimiza o corte de materiais reduzindo desperdícios
e maximizando o aproveitamento do material.
"""

from .core import CutPlanner
from .models import Material, Part, CutOperation, MaterialCut, Leftover, OptimizationResult, OptimizationRequest

__version__ = "1.0.0"
__author__ = "CutPlanner Team"

__all__ = [
    "CutPlanner",
    "Material", 
    "Part",
    "CutOperation",
    "MaterialCut",
    "Leftover",
    "OptimizationResult",
    "OptimizationRequest"
] 