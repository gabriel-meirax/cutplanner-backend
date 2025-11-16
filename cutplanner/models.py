"""
Modelos de dados para o sistema CutPlanner
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class MaterialType(str, Enum):
    """Tipos de materiais suportados"""
    BAR = "bar"           # Barra/Perfil linear
    SHEET = "sheet"       # Chapa/Placa 2D
    PROFILE = "profile"   # Perfil especial


class PartType(str, Enum):
    """Tipos de peças"""
    LINEAR = "linear"     # Peça linear (1D)
    RECTANGULAR = "rectangular"  # Peça retangular (2D)


class Material(BaseModel):
    """Representa um material disponível em estoque"""
    id: str = Field(..., description="Identificador único do material")
    name: str = Field(..., description="Nome descritivo do material")
    material_type: MaterialType = Field(..., description="Tipo do material")
    length: float = Field(..., description="Comprimento (mm) - para materiais 1D")
    width: Optional[float] = Field(None, description="Largura (mm) - para materiais 2D")
    thickness: Optional[float] = Field(None, description="Espessura (mm)")
    quantity: int = Field(..., ge=1, description="Quantidade disponível")
    cost_per_unit: Optional[float] = Field(None, description="Custo por unidade")
    
    @validator('width')
    def validate_width(cls, v, values):
        if values.get('material_type') == MaterialType.SHEET and v is None:
            raise ValueError("Largura é obrigatória para chapas")
        return v
    
    @property
    def area(self) -> float:
        """Área total do material (para 2D)"""
        if self.material_type == MaterialType.SHEET:
            return self.length * self.width
        return self.length
    
    @property
    def total_length(self) -> float:
        """Comprimento total disponível"""
        return self.length * self.quantity


class Part(BaseModel):
    """Representa uma peça a ser cortada"""
    id: str = Field(..., description="Identificador único da peça")
    name: str = Field(..., description="Nome descritivo da peça")
    part_type: PartType = Field(..., description="Tipo da peça")
    length: float = Field(..., description="Comprimento (mm)")
    width: Optional[float] = Field(None, description="Largura (mm) - para peças 2D")
    quantity: int = Field(..., ge=1, description="Quantidade necessária")
    priority: int = Field(1, ge=1, le=10, description="Prioridade de corte (1-10)")
    
    @validator('width')
    def validate_width(cls, v, values):
        if values.get('part_type') == PartType.RECTANGULAR and v is None:
            raise ValueError("Largura é obrigatória para peças retangulares")
        return v
    
    @property
    def area(self) -> float:
        """Área da peça"""
        if self.part_type == PartType.RECTANGULAR:
            return self.length * self.width
        return self.length
    
    @property
    def total_area(self) -> float:
        """Área total necessária"""
        return self.area * self.quantity


class CutOperation(BaseModel):
    """Representa uma operação de corte individual"""
    part_id: str = Field(..., description="ID da peça cortada")
    part_name: str = Field(..., description="Nome da peça")
    position_x: float = Field(0, description="Posição X (mm)")
    position_y: float = Field(0, description="Posição Y (mm) - para 2D")
    length: float = Field(..., description="Comprimento da peça")
    width: Optional[float] = Field(None, description="Largura da peça")
    rotation: float = Field(0, description="Rotação em graus")
    order: int = Field(..., description="Ordem de execução")


class MaterialCut(BaseModel):
    """Representa o corte de um material específico"""
    material_id: str = Field(..., description="ID do material")
    material_name: str = Field(..., description="Nome do material")
    cuts: List[CutOperation] = Field(..., description="Lista de cortes")
    waste: float = Field(..., description="Desperdício total (mm)")
    efficiency: float = Field(..., description="Aproveitamento percentual")
    remaining_length: float = Field(..., description="Comprimento restante")
    remaining_width: Optional[float] = Field(None, description="Largura restante")


class Leftover(BaseModel):
    """Representa um retalho/sobra de material"""
    length: float = Field(..., description="Comprimento do retalho")
    width: Optional[float] = Field(None, description="Largura do retalho")
    material_id: str = Field(..., description="ID do material de origem")
    usable: bool = Field(..., description="Se o retalho é utilizável")
    area: float = Field(..., description="Área do retalho")


class OptimizationResult(BaseModel):
    """Resultado completo da otimização"""
    success: bool = Field(..., description="Se a otimização foi bem-sucedida")
    efficiency: float = Field(..., description="Aproveitamento percentual total")
    total_waste: float = Field(..., description="Desperdício total (mm)")
    total_cost: Optional[float] = Field(None, description="Custo total estimado")
    materials_used: int = Field(..., description="Quantidade de materiais utilizados")
    cuts: List[MaterialCut] = Field(..., description="Lista de cortes por material")
    leftovers: List[Leftover] = Field(..., description="Lista de retalhos")
    execution_order: List[str] = Field(..., description="Ordem de execução dos cortes")
    algorithm_used: str = Field(..., description="Algoritmo utilizado")
    processing_time: float = Field(..., description="Tempo de processamento (ms)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2)
        }


class OptimizationRequest(BaseModel):
    """Requisição para otimização"""
    materials: List[Material] = Field(..., description="Lista de materiais disponíveis")
    parts: List[Part] = Field(..., description="Lista de peças a cortar")
    kerf_width: float = Field(3.0, ge=0, description="Espessura do corte (mm)")
    algorithm: str = Field("best_fit", description="Algoritmo de otimização")
    max_iterations: int = Field(1000, ge=1, description="Máximo de iterações")
    optimize_cost: bool = Field(False, description="Otimizar por custo")
    allow_rotation: bool = Field(True, description="Permitir rotação de peças") 