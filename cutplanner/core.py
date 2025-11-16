"""
Núcleo do sistema CutPlanner com algoritmos de otimização
"""

import time
import random
from typing import List, Dict, Tuple, Optional, Any
from copy import deepcopy
import numpy as np

from .models import (
    Material, Part, CutOperation, MaterialCut, 
    Leftover, OptimizationResult, OptimizationRequest
)


class CutPlanner:
    """
    Sistema principal de otimização de cortes
    """
    
    def __init__(self, kerf_width: float = 3.0):
        """
        Inicializa o planejador de cortes
        
        Args:
            kerf_width: Espessura do corte em mm
        """
        self.kerf_width = kerf_width
        self.algorithms = {
            "first_fit": self._first_fit_1d,
            "best_fit": self._best_fit_1d,
            "genetic": self._genetic_algorithm_1d,
            "guillotine": self._guillotine_2d,
            "maxrects": self._maxrects_2d
        }
    
    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Otimiza o corte baseado no tipo de material
        
        Args:
            request: Requisição de otimização
            
        Returns:
            Resultado da otimização
        """
        start_time = time.time()
        
        try:
            # Determinar tipo de otimização
            if any(m.material_type.value in ["bar", "profile"] for m in request.materials):
                result = self._optimize_1d(request)
            else:
                result = self._optimize_2d(request)
            
            # Calcular tempo de processamento
            processing_time = (time.time() - start_time) * 1000
            
            # Atualizar metadados
            result.processing_time = processing_time
            result.algorithm_used = request.algorithm
            
            return result
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                efficiency=0.0,
                total_waste=0.0,
                materials_used=0,
                cuts=[],
                leftovers=[],
                execution_order=[],
                algorithm_used=request.algorithm,
                processing_time=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    def _optimize_1d(self, request: OptimizationRequest) -> OptimizationResult:
        """Otimização para materiais 1D (barras/perfis)"""
        
        # Preparar dados
        materials = self._prepare_materials_1d(request.materials)
        parts = self._prepare_parts_1d(request.parts)
        
        # Executar algoritmo selecionado
        if request.algorithm in self.algorithms:
            cuts, leftovers = self.algorithms[request.algorithm](
                materials, parts, request.kerf_width
            )
        else:
            # Fallback para best_fit
            cuts, leftovers = self._best_fit_1d(materials, parts, request.kerf_width)
        
        # Calcular métricas
        total_waste = sum(cut.waste for cut in cuts)
        total_area = sum(m.area for m in request.materials)
        used_area = total_area - total_waste
        efficiency = (used_area / total_area) * 100 if total_area > 0 else 0
        
        # Gerar ordem de execução
        execution_order = self._generate_execution_order(cuts)
        
        return OptimizationResult(
            success=True,
            efficiency=efficiency,
            total_waste=total_waste,
            materials_used=len(cuts),
            cuts=cuts,
            leftovers=leftovers,
            execution_order=execution_order,
            algorithm_used=request.algorithm,
            processing_time=0,  # Será atualizado pelo método principal
            metadata={"dimension": "1D"}
        )
    
    def _optimize_2d(self, request: OptimizationRequest) -> OptimizationResult:
        """Otimização para materiais 2D (chapas)"""
        
        # Preparar dados
        materials = self._prepare_materials_2d(request.materials)
        parts = self._prepare_parts_2d(request.parts)
        
        # Executar algoritmo selecionado
        if request.algorithm in ["guillotine", "maxrects"]:
            cuts, leftovers = self.algorithms[request.algorithm](
                materials, parts, request.kerf_width
            )
        else:
            # Fallback para guillotine
            cuts, leftovers = self._guillotine_2d(materials, parts, request.kerf_width)
        
        # Calcular métricas
        total_waste = sum(cut.waste for cut in cuts)
        total_area = sum(m.area for m in request.materials)
        used_area = total_area - total_waste
        efficiency = (used_area / total_area) * 100 if total_area > 0 else 0
        
        # Gerar ordem de execução
        execution_order = self._generate_execution_order(cuts)
        
        return OptimizationResult(
            success=True,
            efficiency=efficiency,
            total_waste=total_waste,
            materials_used=len(cuts),
            cuts=cuts,
            leftovers=leftovers,
            execution_order=execution_order,
            algorithm_used=request.algorithm,
            processing_time=0,  # Será atualizado pelo método principal
            metadata={"dimension": "2D"}
        )
    
    def _prepare_materials_1d(self, materials: List[Material]) -> List[Dict]:
        """Prepara materiais 1D para processamento"""
        prepared = []
        for material in materials:
            if material.material_type.value in ["bar", "profile"]:
                for i in range(material.quantity):
                    prepared.append({
                        "id": f"{material.id}_{i+1}",
                        "original_id": material.id,
                        "name": material.name,
                        "length": material.length,
                        "cost": material.cost_per_unit or 0,
                        "used": False
                    })
        return prepared
    
    def _prepare_parts_1d(self, parts: List[Part]) -> List[Dict]:
        """Prepara peças 1D para processamento"""
        prepared = []
        for part in parts:
            if part.part_type.value == "linear":
                for i in range(part.quantity):
                    prepared.append({
                        "id": f"{part.id}_{i+1}",
                        "original_id": part.id,
                        "name": part.name,
                        "length": part.length,
                        "priority": part.priority,
                        "assigned": False
                    })
        return prepared
    
    def _prepare_materials_2d(self, materials: List[Material]) -> List[Dict]:
        """Prepara materiais 2D para processamento"""
        prepared = []
        for material in materials:
            if material.material_type.value == "sheet":
                for i in range(material.quantity):
                    prepared.append({
                        "id": f"{material.id}_{i+1}",
                        "original_id": material.id,
                        "name": material.name,
                        "width": material.width,
                        "height": material.length,
                        "cost": material.cost_per_unit or 0,
                        "used": False,
                        "cuts": []
                    })
        return prepared
    
    def _prepare_parts_2d(self, parts: List[Part]) -> List[Dict]:
        """Prepara peças 2D para processamento"""
        prepared = []
        for part in parts:
            if part.part_type.value == "rectangular":
                for i in range(part.quantity):
                    prepared.append({
                        "id": f"{part.id}_{i+1}",
                        "original_id": part.id,
                        "name": part.name,
                        "width": part.width,
                        "height": part.length,
                        "priority": part.priority,
                        "assigned": False
                    })
        return prepared
    
    def _first_fit_1d(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Algoritmo First Fit para otimização 1D"""
        
        # Ordenar peças por prioridade e tamanho (maior primeiro)
        parts.sort(key=lambda x: (x["priority"], x["length"]), reverse=True)
        
        cuts = []
        leftovers = []
        
        # Criar um dicionário para rastrear materiais disponíveis
        available_materials = {mat["id"]: mat.copy() for mat in materials}
        
        for part in parts:
            if part["assigned"]:
                continue
                
            # Procurar primeiro material que caiba
            material_found = False
            
            # Primeiro, tentar encontrar um material já usado que tenha espaço
            for cut in cuts:
                if cut.remaining_length >= part["length"] + kerf_width:
                    # Adicionar corte neste material
                    cut_op = CutOperation(
                        part_id=part["id"],
                        part_name=part["name"],
                        position_x=cut.remaining_length - cut.waste,
                        length=part["length"],
                        order=len(cut.cuts) + 1
                    )
                    cut.cuts.append(cut_op)
                    
                    # Atualizar desperdício e comprimento restante
                    cut.waste -= (part["length"] + kerf_width)
                    cut.remaining_length = cut.waste
                    cut.efficiency = ((available_materials[cut.material_id]["length"] - cut.waste) / available_materials[cut.material_id]["length"]) * 100
                    
                    part["assigned"] = True
                    material_found = True
                    break
            
            # Se não encontrou material usado, procurar um novo
            if not material_found:
                for material_id, material in available_materials.items():
                    if material["used"]:
                        continue
                        
                    # Verificar se a peça cabe neste material
                    if part["length"] + kerf_width <= material["length"]:
                        # Criar novo corte para este material
                        material_cut = MaterialCut(
                            material_id=material["id"],
                            material_name=material["name"],
                            cuts=[],
                            waste=material["length"] - (part["length"] + kerf_width),
                            efficiency=((part["length"] + kerf_width) / material["length"]) * 100,
                            remaining_length=material["length"] - (part["length"] + kerf_width)
                        )
                        
                        # Adicionar corte
                        cut_op = CutOperation(
                            part_id=part["id"],
                            part_name=part["name"],
                            position_x=0,
                            length=part["length"],
                            order=1
                        )
                        material_cut.cuts.append(cut_op)
                        
                        cuts.append(material_cut)
                        material["used"] = True
                        part["assigned"] = True
                        break
        
        # Processar retalhos
        for cut in cuts:
            if cut.waste > 50:  # Retalhos maiores que 50mm são considerados utilizáveis
                leftover = Leftover(
                    length=cut.waste,
                    material_id=cut.material_id,
                    usable=True,
                    area=cut.waste
                )
                leftovers.append(leftover)
        
        return cuts, leftovers
    
    def _best_fit_1d(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Algoritmo Best Fit para otimização 1D"""
        
        # Ordenar peças por prioridade e tamanho (maior primeiro)
        parts.sort(key=lambda x: (x["priority"], x["length"]), reverse=True)
        
        cuts = []
        leftovers = []
        
        # Criar um dicionário para rastrear materiais disponíveis
        available_materials = {mat["id"]: mat.copy() for mat in materials}
        
        for part in parts:
            if part["assigned"]:
                continue
                
            # Encontrar o melhor material para esta peça
            best_material_id = None
            best_waste = float('inf')
            best_position = 0
            
            # Primeiro, tentar encontrar um material já usado que tenha espaço
            for cut in cuts:
                if cut.remaining_length >= part["length"] + kerf_width:
                    waste = cut.remaining_length - (part["length"] + kerf_width)
                    if waste < best_waste:
                        best_waste = waste
                        best_material_id = cut.material_id
                        # Calcular posição baseada no comprimento usado
                        used_length = available_materials[cut.material_id]["length"] - cut.remaining_length
                        best_position = used_length
            
            # Se não encontrou material usado, procurar um novo
            if best_material_id is None:
                for material_id, material in available_materials.items():
                    if material["used"]:
                        continue
                        
                    # Calcular desperdício se a peça for colocada neste material
                    if part["length"] + kerf_width <= material["length"]:
                        waste = material["length"] - (part["length"] + kerf_width)
                        if waste < best_waste:
                            best_waste = waste
                            best_material_id = material_id
                            best_position = 0
            
            if best_material_id:
                # Encontrar ou criar o corte do material
                material_cut = None
                for cut in cuts:
                    if cut.material_id == best_material_id:
                        material_cut = cut
                        break
                
                if not material_cut:
                    # Criar novo corte para este material
                    material = available_materials[best_material_id]
                    material_cut = MaterialCut(
                        material_id=material["id"],
                        material_name=material["name"],
                        cuts=[],
                        waste=material["length"] - (part["length"] + kerf_width),
                        efficiency=((part["length"] + kerf_width) / material["length"]) * 100,
                        remaining_length=material["length"] - (part["length"] + kerf_width)
                    )
                    cuts.append(material_cut)
                    material["used"] = True
                
                # Adicionar corte
                cut_op = CutOperation(
                    part_id=part["id"],
                    part_name=part["name"],
                    position_x=best_position,
                    length=part["length"],
                    order=len(material_cut.cuts) + 1
                )
                material_cut.cuts.append(cut_op)
                
                # Atualizar desperdício e eficiência
                material_cut.waste = best_waste
                material_cut.efficiency = ((available_materials[best_material_id]["length"] - best_waste) / available_materials[best_material_id]["length"]) * 100
                material_cut.remaining_length = best_waste
                
                part["assigned"] = True
        
        # Processar retalhos
        for cut in cuts:
            if cut.waste > 50:  # Retalhos maiores que 50mm são considerados utilizáveis
                leftover = Leftover(
                    length=cut.waste,
                    material_id=cut.material_id,
                    usable=True,
                    area=cut.waste
                )
                leftovers.append(leftover)
        
        return cuts, leftovers
    
    def _genetic_algorithm_1d(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Algoritmo Genético para otimização 1D"""
        
        # Implementação simplificada do algoritmo genético
        # Em produção, seria mais sofisticado
        
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # Criar população inicial
        population = []
        for _ in range(population_size):
            individual = self._create_random_solution(materials, parts, kerf_width)
            population.append(individual)
        
        # Evolução
        for generation in range(generations):
            # Avaliar fitness
            fitness_scores = [self._calculate_fitness(individual) for individual in population]
            
            # Seleção e reprodução
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                
                # Mutação
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Retornar melhor solução
        best_individual = max(population, key=self._calculate_fitness)
        return self._convert_solution_to_cuts(best_individual, materials, parts, kerf_width)
    
    def _create_random_solution(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> List[Dict]:
        """Cria uma solução aleatória para o algoritmo genético"""
        solution = []
        for part in parts:
            if not part["assigned"]:
                material = random.choice([m for m in materials if not m["used"]])
                solution.append({
                    "part_id": part["id"],
                    "material_id": material["id"],
                    "position": random.randint(0, int(material["length"] - part["length"]))
                })
        return solution
    
    def _calculate_fitness(self, individual: List[Dict]) -> float:
        """Calcula o fitness de uma solução"""
        # Implementação simplificada - em produção seria mais complexa
        return random.random()
    
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> Dict:
        """Seleção por torneio"""
        tournament_size = 3
        tournament = random.sample(list(enumerate(population)), tournament_size)
        winner_idx = max(tournament, key=lambda x: fitness_scores[x[0]])[0]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """Operador de crossover"""
        # Implementação simplificada
        return parent1 if random.random() < 0.5 else parent2
    
    def _mutate(self, individual: List[Dict]) -> List[Dict]:
        """Operador de mutação"""
        # Implementação simplificada
        return individual
    
    def _convert_solution_to_cuts(self, solution: List[Dict], materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Converte solução genética para formato de cortes"""
        # Implementação simplificada - retorna solução básica
        return self._best_fit_1d(materials, parts, kerf_width)
    
    def _guillotine_2d(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Algoritmo de corte guilhotina para materiais 2D"""
        # Implementação simplificada - em produção seria mais complexa
        cuts = []
        leftovers = []
        
        # Por enquanto, retorna solução básica
        for material in materials:
            if not material["used"]:
                material_cut = MaterialCut(
                    material_id=material["id"],
                    material_name=material["name"],
                    cuts=[],
                    waste=material["width"] * material["height"],
                    efficiency=0,
                    remaining_length=material["height"],
                    remaining_width=material["width"]
                )
                cuts.append(material_cut)
                material["used"] = True
        
        return cuts, leftovers
    
    def _maxrects_2d(self, materials: List[Dict], parts: List[Dict], kerf_width: float) -> Tuple[List[MaterialCut], List[Leftover]]:
        """Algoritmo MaxRects para materiais 2D"""
        # Implementação simplificada - em produção seria mais complexa
        return self._guillotine_2d(materials, parts, kerf_width)
    
    def _generate_execution_order(self, cuts: List[MaterialCut]) -> List[str]:
        """Gera ordem de execução dos cortes"""
        execution_order = []
        
        for material_cut in cuts:
            for cut_op in material_cut.cuts:
                execution_order.append(f"{cut_op.part_name} em {material_cut.material_name}")
        
        return execution_order
    
    def optimize_1d(self, materials: List[Material], parts: List[Part], **kwargs) -> OptimizationResult:
        """Método de conveniência para otimização 1D"""
        request = OptimizationRequest(
            materials=materials,
            parts=parts,
            **kwargs
        )
        return self.optimize(request)
    
    def optimize_2d(self, materials: List[Material], parts: List[Part], **kwargs) -> OptimizationResult:
        """Método de conveniência para otimização 2D"""
        request = OptimizationRequest(
            materials=materials,
            parts=parts,
            **kwargs
        )
        return self.optimize(request) 