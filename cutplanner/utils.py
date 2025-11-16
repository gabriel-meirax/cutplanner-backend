"""
Utilitários para visualização e relatórios do CutPlanner
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .models import OptimizationResult, MaterialCut, CutOperation, Leftover


class CutPlannerVisualizer:
    """Classe para visualização dos resultados de otimização"""
    
    def __init__(self, result: OptimizationResult):
        """
        Inicializa o visualizador
        
        Args:
            result: Resultado da otimização
        """
        self.result = result
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    def plot_1d_cuts(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Plota visualização 1D dos cortes"""
        if not self.result.cuts:
            print("Nenhum corte para visualizar")
            return
        
        fig, axes = plt.subplots(len(self.result.cuts), 1, figsize=(12, 4 * len(self.result.cuts)))
        if len(self.result.cuts) == 1:
            axes = [axes]
        
        for i, material_cut in enumerate(self.result.cuts):
            ax = axes[i]
            
            # Configurar eixo
            ax.set_xlim(0, self._get_material_length(material_cut))
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(f"{material_cut.material_name} - Eficiência: {material_cut.efficiency:.1f}%")
            ax.set_xlabel("Posição (mm)")
            ax.axhline(y=0, color='black', linewidth=2)
            
            # Plotar cortes
            for j, cut_op in enumerate(material_cut.cuts):
                color = self.colors[j % len(self.colors)]
                
                # Retângulo representando a peça
                rect = Rectangle((cut_op.position_x, -0.2), cut_op.length, 0.4, 
                               facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Texto com informações da peça
                ax.text(cut_op.position_x + cut_op.length/2, 0.3, 
                       f"{cut_op.part_name}\n{cut_op.length}mm", 
                       ha='center', va='center', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Linha de corte
                if j < len(material_cut.cuts) - 1:
                    next_cut = material_cut.cuts[j + 1]
                    cut_position = cut_op.position_x + cut_op.length
                    ax.axvline(x=cut_position, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax.text(cut_position, 0.4, f"Corte {j+1}", 
                           ha='center', va='bottom', fontsize=8, color='red')
            
            # Marcar desperdício
            if material_cut.waste > 0:
                waste_start = sum(cut.length for cut in material_cut.cuts) + len(material_cut.cuts) * 3  # 3mm kerf
                ax.axvspan(waste_start, self._get_material_length(material_cut), 
                          alpha=0.3, color='red', label=f'Desperdício: {material_cut.waste:.1f}mm')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_2d_cuts(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Plota visualização 2D dos cortes"""
        if not self.result.cuts:
            print("Nenhum corte para visualizar")
            return
        
        fig, axes = plt.subplots(1, len(self.result.cuts), figsize=(6 * len(self.result.cuts), 6))
        if len(self.result.cuts) == 1:
            axes = [axes]
        
        for i, material_cut in enumerate(self.result.cuts):
            ax = axes[i]
            
            # Configurar eixo
            material_width = self._get_material_width(material_cut)
            material_height = self._get_material_length(material_cut)
            
            ax.set_xlim(0, material_width)
            ax.set_ylim(0, material_height)
            ax.set_title(f"{material_cut.material_name}\nEficiência: {material_cut.efficiency:.1f}%")
            ax.set_xlabel("Largura (mm)")
            ax.set_ylabel("Altura (mm)")
            ax.grid(True, alpha=0.3)
            
            # Plotar material base
            rect = Rectangle((0, 0), material_width, material_height, 
                           facecolor='lightgray', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Plotar cortes
            for j, cut_op in enumerate(material_cut.cuts):
                color = self.colors[j % len(self.colors)]
                
                # Retângulo representando a peça
                if hasattr(cut_op, 'width') and cut_op.width:
                    rect = Rectangle((cut_op.position_x, cut_op.position_y), 
                                   cut_op.width, cut_op.length,
                                   facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    
                    # Texto com informações da peça
                    ax.text(cut_op.position_x + cut_op.width/2, 
                           cut_op.position_y + cut_op.length/2,
                           f"{cut_op.part_name}\n{cut_op.width}x{cut_op.length}mm", 
                           ha='center', va='center', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def _get_material_length(self, material_cut: MaterialCut) -> float:
        """Obtém o comprimento total do material"""
        total_cuts_length = sum(cut.length for cut in material_cut.cuts)
        total_kerf = len(material_cut.cuts) * 3  # 3mm kerf
        return total_cuts_length + total_kerf + material_cut.waste
    
    def _get_material_width(self, material_cut: MaterialCut) -> float:
        """Obtém a largura do material (para 2D)"""
        # Implementação simplificada - em produção seria mais robusta
        return 1000  # 1m padrão
    
    def create_summary_chart(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Cria gráfico de resumo da otimização"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Eficiência por material
        material_names = [cut.material_name for cut in self.result.cuts]
        efficiencies = [cut.efficiency for cut in self.result.cuts]
        
        bars1 = ax1.bar(material_names, efficiencies, color='skyblue', edgecolor='navy')
        ax1.set_title('Eficiência por Material')
        ax1.set_ylabel('Eficiência (%)')
        ax1.set_ylim(0, 100)
        
        # Adicionar valores nas barras
        for bar, eff in zip(bars1, efficiencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{eff:.1f}%', ha='center', va='bottom')
        
        # Gráfico 2: Distribuição de desperdício
        waste_values = [cut.waste for cut in self.result.cuts]
        ax2.pie(waste_values, labels=material_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribuição do Desperdício')
        
        # Gráfico 3: Comparação de eficiência vs desperdício
        ax3.scatter(efficiencies, waste_values, s=100, alpha=0.7)
        ax3.set_xlabel('Eficiência (%)')
        ax3.set_ylabel('Desperdício (mm)')
        ax3.set_title('Eficiência vs Desperdício')
        ax3.grid(True, alpha=0.3)
        
        # Adicionar labels dos materiais
        for i, name in enumerate(material_names):
            ax3.annotate(name, (efficiencies[i], waste_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Gráfico 4: Resumo geral
        ax4.axis('off')
        summary_text = f"""
        RESUMO DA OTIMIZAÇÃO
        
        Eficiência Total: {self.result.efficiency:.1f}%
        Desperdício Total: {self.result.total_waste:.1f} mm
        Materiais Utilizados: {self.result.materials_used}
        Algoritmo: {self.result.algorithm_used}
        Tempo de Processamento: {self.result.processing_time:.1f} ms
        
        Retalhos Utilizáveis: {len([l for l in self.result.leftovers if l.usable])}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()


class CutPlannerReporter:
    """Classe para geração de relatórios"""
    
    def __init__(self, result: OptimizationResult):
        """
        Inicializa o gerador de relatórios
        
        Args:
            result: Resultado da otimização
        """
        self.result = result
    
    def generate_text_report(self) -> str:
        """Gera relatório em formato texto"""
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE OTIMIZAÇÃO DE CORTES")
        report.append("=" * 60)
        report.append("")
        
        # Resumo geral
        report.append("RESUMO GERAL:")
        report.append(f"  • Eficiência Total: {self.result.efficiency:.1f}%")
        report.append(f"  • Desperdício Total: {self.result.total_waste:.1f} mm")
        report.append(f"  • Materiais Utilizados: {self.result.materials_used}")
        report.append(f"  • Algoritmo Utilizado: {self.result.algorithm_used}")
        report.append(f"  • Tempo de Processamento: {self.result.processing_time:.1f} ms")
        report.append("")
        
        # Detalhes por material
        report.append("DETALHES POR MATERIAL:")
        report.append("-" * 40)
        
        for i, material_cut in enumerate(self.result.cuts, 1):
            report.append(f"\n{i}. {material_cut.material_name}:")
            report.append(f"   • Eficiência: {material_cut.efficiency:.1f}%")
            report.append(f"   • Desperdício: {material_cut.waste:.1f} mm")
            report.append(f"   • Peças cortadas: {len(material_cut.cuts)}")
            
            for j, cut_op in enumerate(material_cut.cuts, 1):
                report.append(f"     {j}. {cut_op.part_name}: {cut_op.length}mm (pos: {cut_op.position_x}mm)")
        
        # Retalhos
        if self.result.leftovers:
            report.append("\nRETALHOS UTILIZÁVEIS:")
            report.append("-" * 30)
            for leftover in self.result.leftovers:
                if leftover.usable:
                    report.append(f"  • {leftover.length:.1f}mm (Material: {leftover.material_id})")
        
        # Ordem de execução
        report.append("\nORDEM DE EXECUÇÃO:")
        report.append("-" * 25)
        for i, step in enumerate(self.result.execution_order, 1):
            report.append(f"  {i}. {step}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def generate_csv_report(self, file_path: str) -> None:
        """Gera relatório em formato CSV"""
        # Dados dos cortes
        cuts_data = []
        for material_cut in self.result.cuts:
            for cut_op in material_cut.cuts:
                cuts_data.append({
                    'Material': material_cut.material_name,
                    'Peça': cut_op.part_name,
                    'Comprimento': cut_op.length,
                    'Posição_X': cut_op.position_x,
                    'Posição_Y': getattr(cut_op, 'position_y', 0),
                    'Ordem': cut_op.order,
                    'Eficiência_Material': material_cut.efficiency,
                    'Desperdício_Material': material_cut.waste
                })
        
        # Dados dos materiais
        materials_data = []
        for material_cut in self.result.cuts:
            materials_data.append({
                'Material': material_cut.material_name,
                'Eficiência': material_cut.efficiency,
                'Desperdício': material_cut.waste,
                'Peças_Cortadas': len(material_cut.cuts),
                'Comprimento_Restante': material_cut.remaining_length
            })
        
        # Dados dos retalhos
        leftovers_data = []
        for leftover in self.result.leftovers:
            leftovers_data.append({
                'Material_Origem': leftover.material_id,
                'Comprimento': leftover.length,
                'Largura': leftover.width or 0,
                'Área': leftover.area,
                'Utilizável': leftover.usable
            })
        
        # Salvar arquivos CSV
        with open(f"{file_path}_cortes.csv", 'w', newline='', encoding='utf-8') as f:
            if cuts_data:
                writer = csv.DictWriter(f, fieldnames=cuts_data[0].keys())
                writer.writeheader()
                writer.writerows(cuts_data)
        
        with open(f"{file_path}_materiais.csv", 'w', newline='', encoding='utf-8') as f:
            if materials_data:
                writer = csv.DictWriter(f, fieldnames=materials_data[0].keys())
                writer.writeheader()
                writer.writerows(materials_data)
        
        with open(f"{file_path}_retalhos.csv", 'w', newline='', encoding='utf-8') as f:
            if leftovers_data:
                writer = csv.DictWriter(f, fieldnames=leftovers_data[0].keys())
                writer.writeheader()
                writer.writerows(leftovers_data)
    
    def generate_json_report(self, file_path: str) -> None:
        """Gera relatório em formato JSON"""
        # Converter para dict para serialização
        report_data = self.result.dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    def generate_html_report(self, file_path: str) -> str:
        """Gera relatório em formato HTML"""
        html = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório de Otimização - CutPlanner</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .material {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .cuts {{ margin-left: 20px; }}
                .cut-item {{ background-color: white; padding: 8px; margin: 5px 0; border-radius: 3px; }}
                .leftovers {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .execution-order {{ background-color: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔧 CutPlanner - Relatório de Otimização</h1>
                <p>Relatório gerado automaticamente pelo sistema de otimização de cortes</p>
            </div>
            
            <div class="summary">
                <h2>📊 Resumo Geral</h2>
                <div class="metric">
                    <div class="metric-value">{self.result.efficiency:.1f}%</div>
                    <div class="metric-label">Eficiência Total</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.result.total_waste:.1f}mm</div>
                    <div class="metric-label">Desperdício Total</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.result.materials_used}</div>
                    <div class="metric-label">Materiais Utilizados</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.result.processing_time:.1f}ms</div>
                    <div class="metric-label">Tempo de Processamento</div>
                </div>
                <p><strong>Algoritmo Utilizado:</strong> {self.result.algorithm_used}</p>
            </div>
            
            <h2>📦 Detalhes por Material</h2>
        """
        
        for i, material_cut in enumerate(self.result.cuts, 1):
            html += f"""
            <div class="material">
                <h3>{i}. {material_cut.material_name}</h3>
                <p><strong>Eficiência:</strong> {material_cut.efficiency:.1f}% | 
                   <strong>Desperdício:</strong> {material_cut.waste:.1f}mm | 
                   <strong>Peças:</strong> {len(material_cut.cuts)}</p>
                
                <div class="cuts">
                    <h4>Peças Cortadas:</h4>
            """
            
            for j, cut_op in enumerate(material_cut.cuts, 1):
                html += f"""
                    <div class="cut-item">
                        <strong>{j}.</strong> {cut_op.part_name} - {cut_op.length}mm 
                        (Posição: {cut_op.position_x}mm, Ordem: {cut_op.order})
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        if self.result.leftovers:
            html += """
            <div class="leftovers">
                <h2>♻️ Retalhos Utilizáveis</h2>
            """
            
            for leftover in self.result.leftovers:
                if leftover.usable:
                    html += f"""
                    <p>• <strong>{leftover.length:.1f}mm</strong> (Material: {leftover.material_id})</p>
                    """
            
            html += "</div>"
        
        html += f"""
            <div class="execution-order">
                <h2>📋 Ordem de Execução</h2>
                <ol>
        """
        
        for step in self.result.execution_order:
            html += f"<li>{step}</li>"
        
        html += """
                </ol>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                <p>Relatório gerado pelo CutPlanner - Sistema de Otimização de Cortes</p>
                <p>Data: """ + str(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")) + """</p>
            </div>
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return html


def export_result(result: OptimizationResult, output_dir: str, formats: List[str] = None) -> None:
    """
    Exporta resultado em múltiplos formatos
    
    Args:
        result: Resultado da otimização
        output_dir: Diretório de saída
        formats: Lista de formatos (txt, csv, json, html)
    """
    if formats is None:
        formats = ["txt", "csv", "json", "html"]
    
    # Criar diretório se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    reporter = CutPlannerReporter(result)
    base_path = Path(output_dir) / "relatorio_cutplanner"
    
    if "txt" in formats:
        report_text = reporter.generate_text_report()
        with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    if "csv" in formats:
        reporter.generate_csv_report(str(base_path))
    
    if "json" in formats:
        reporter.generate_json_report(f"{base_path}.json")
    
    if "html" in formats:
        reporter.generate_html_report(f"{base_path}.html")
    
    print(f"Relatórios exportados para: {output_dir}")


def create_visualization(result: OptimizationResult, output_dir: str, show: bool = False) -> None:
    """
    Cria visualizações do resultado
    
    Args:
        result: Resultado da otimização
        output_dir: Diretório de saída
        show: Se deve mostrar os gráficos
    """
    # Criar diretório se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    visualizer = CutPlannerVisualizer(result)
    base_path = Path(output_dir) / "visualizacao_cutplanner"
    
    # Determinar tipo de visualização baseado no resultado
    if result.metadata.get("dimension") == "2D":
        visualizer.plot_2d_cuts(f"{base_path}_2d.png", show=show)
    else:
        visualizer.plot_1d_cuts(f"{base_path}_1d.png", show=show)
    
    # Gráfico de resumo
    visualizer.create_summary_chart(f"{base_path}_resumo.png", show=show)
    
    print(f"Visualizações salvas em: {output_dir}") 