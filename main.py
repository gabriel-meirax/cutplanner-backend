"""
Servidor FastAPI principal para o CutPlanner
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Adicionar o diretório raiz ao path para importar cutplanner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutplanner import CutPlanner
from cutplanner.models import OptimizationRequest, OptimizationResult
from cutplanner.utils import CutPlannerReporter, CutPlannerVisualizer
import tempfile
from pathlib import Path

# Configuração do FastAPI
app = FastAPI(
    title="CutPlanner API",
    description="API para otimização de cortes de materiais para serralherias",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instância global do CutPlanner
cut_planner = CutPlanner()


@app.get("/")
async def root():
    """Página inicial da API - redireciona para documentação"""
    return {
        "message": "CutPlanner API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "service": "CutPlanner API",
        "version": "1.0.0"
    }


@app.post("/optimize/1d")
async def optimize_1d(request: OptimizationRequest):
    """
    Otimização 1D para barras e perfis
    
    Args:
        request: Requisição de otimização
        
    Returns:
        Resultado da otimização em formato JSON
    """
    try:
        # Validar se os materiais são 1D
        for material in request.materials:
            if material.material_type.value not in ["bar", "profile"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Material {material.name} não é 1D. Use 'bar' ou 'profile'."
                )
        
        # Executar otimização
        result = cut_planner.optimize(request)
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na otimização: {result.metadata.get('error', 'Erro desconhecido')}"
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/2d")
async def optimize_2d(request: OptimizationRequest):
    """
    Otimização 2D para chapas e placas
    
    Args:
        request: Requisição de otimização
        
    Returns:
        Resultado da otimização em formato JSON
    """
    try:
        # Validar se os materiais são 2D
        for material in request.materials:
            if material.material_type.value != "sheet":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Material {material.name} não é 2D. Use 'sheet'."
                )
        
        # Executar otimização
        result = cut_planner.optimize(request)
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na otimização: {result.metadata.get('error', 'Erro desconhecido')}"
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/auto")
async def optimize_auto(request: OptimizationRequest):
    """
    Otimização automática (detecta tipo baseado nos materiais)
    
    Args:
        request: Requisição de otimização
        
    Returns:
        Resultado da otimização em formato JSON
    """
    try:
        result = cut_planner.optimize(request)
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na otimização: {result.metadata.get('error', 'Erro desconhecido')}"
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/batch")
async def optimize_batch(requests: list[OptimizationRequest]):
    """
    Otimização em lote de múltiplas requisições
    
    Args:
        requests: Lista de requisições de otimização
        
    Returns:
        Lista de resultados da otimização
    """
    try:
        results = []
        for request in requests:
            result = cut_planner.optimize(request)
            results.append(result)
        
        return {
            "total_requests": len(requests),
            "successful": len([r for r in results if r.success]),
            "failed": len([r for r in results if not r.success]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/algorithms")
async def get_algorithms():
    """Retorna lista de algoritmos disponíveis"""
    return {
        "1d_algorithms": ["first_fit", "best_fit", "genetic"],
        "2d_algorithms": ["guillotine", "maxrects"],
        "default_1d": "best_fit",
        "default_2d": "guillotine"
    }


@app.post("/report/generate")
async def generate_report(optimization_result: OptimizationResult, format: str = "all"):
    """
    Gera relatórios em diferentes formatos
    
    Args:
        optimization_result: Resultado da otimização
        format: Formato do relatório (txt, csv, json, html, all)
        
    Returns:
        Relatório no formato solicitado
    """
    try:
        formats = ["txt", "csv", "json", "html"] if format == "all" else [format]
        
        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = CutPlannerReporter(optimization_result)
            
            results = {}
            
            if "txt" in formats:
                report_text = reporter.generate_text_report()
                results["txt"] = report_text
            
            if "json" in formats:
                results["json"] = optimization_result.dict()
            
            if "csv" in formats:
                # Gerar CSV em diretório temporário
                base_path = Path(temp_dir) / "report"
                reporter.generate_csv_report(str(base_path))
                
                # Ler e retornar conteúdo dos CSVs
                csv_data = {}
                for csv_file in Path(temp_dir).glob("*.csv"):
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        csv_data[csv_file.stem] = f.read()
                results["csv"] = csv_data
            
            if "html" in formats:
                # Gerar HTML em diretório temporário
                base_path = Path(temp_dir) / "report.html"
                html_content = reporter.generate_html_report(str(base_path))
                results["html"] = html_content
            
            return {
                "formats_generated": formats,
                "results": results
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualization/create")
async def create_visualization(optimization_result: OptimizationResult, show: bool = False):
    """
    Cria visualizações do resultado de otimização
    
    Args:
        optimization_result: Resultado da otimização
        show: Se deve mostrar os gráficos
        
    Returns:
        Informações sobre as visualizações criadas
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = CutPlannerVisualizer(optimization_result)
            base_path = Path(temp_dir) / "visualization"
            
            # Determinar tipo de visualização
            if optimization_result.metadata.get("dimension") == "2D":
                visualizer.plot_2d_cuts(f"{base_path}_2d.png", show=show)
            else:
                visualizer.plot_1d_cuts(f"{base_path}_1d.png", show=show)
            
            # Gráfico de resumo
            visualizer.create_summary_chart(f"{base_path}_resumo.png", show=show)
            
            # Retornar informações sobre os arquivos criados
            created_files = list(Path(temp_dir).glob("*.png"))
            
            return {
                "message": "Visualizações criadas com sucesso",
                "files_created": [f.name for f in created_files],
                "temp_directory": temp_dir,
                "note": "Arquivos estão em diretório temporário. Use /download para baixar."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples/1d")
async def get_1d_example():
    """Retorna exemplo de dados para otimização 1D"""
    return {
        "materials": [
            {
                "id": "barra_aco_6m",
                "name": "Barra de Aço 6m",
                "material_type": "bar",
                "length": 6000,
                "quantity": 5,
                "cost_per_unit": 150.0
            },
            {
                "id": "perfil_aluminio_4m",
                "name": "Perfil de Alumínio 4m",
                "material_type": "profile",
                "length": 4000,
                "quantity": 3,
                "cost_per_unit": 120.0
            }
        ],
        "parts": [
            {
                "id": "viga_principal",
                "name": "Viga Principal",
                "part_type": "linear",
                "length": 1200,
                "quantity": 10,
                "priority": 1
            },
            {
                "id": "suporte_secundario",
                "name": "Suporte Secundário",
                "part_type": "linear",
                "length": 800,
                "quantity": 15,
                "priority": 2
            },
            {
                "id": "conector",
                "name": "Conector",
                "part_type": "linear",
                "length": 600,
                "quantity": 20,
                "priority": 3
            }
        ],
        "kerf_width": 3.0,
        "algorithm": "best_fit"
    }


@app.get("/examples/2d")
async def get_2d_example():
    """Retorna exemplo de dados para otimização 2D"""
    return {
        "materials": [
            {
                "id": "chapa_aco_2x3m",
                "name": "Chapa de Aço 2x3m",
                "material_type": "sheet",
                "length": 3000,
                "width": 2000,
                "thickness": 5,
                "quantity": 4,
                "cost_per_unit": 800.0
            }
        ],
        "parts": [
            {
                "id": "painel_frontal",
                "name": "Painel Frontal",
                "part_type": "rectangular",
                "length": 800,
                "width": 600,
                "quantity": 8,
                "priority": 1
            },
            {
                "id": "base_suporte",
                "name": "Base de Suporte",
                "part_type": "rectangular",
                "length": 400,
                "width": 300,
                "quantity": 12,
                "priority": 2
            }
        ],
        "kerf_width": 3.0,
        "algorithm": "guillotine"
    }


@app.get("/stats")
async def get_stats():
    """Retorna estatísticas do sistema"""
    return {
        "service": "CutPlanner API",
        "version": "1.0.0",
        "status": "running",
        "algorithms_supported": {
            "1d": ["first_fit", "best_fit", "genetic"],
            "2d": ["guillotine", "maxrects"]
        },
        "features": [
            "Otimização 1D para barras e perfis",
            "Otimização 2D para chapas",
            "Múltiplos algoritmos",
            "Relatórios em múltiplos formatos",
            "Visualizações gráficas",
            "API REST completa"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 