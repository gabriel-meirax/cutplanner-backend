#!/usr/bin/env python3
"""
Script principal para executar o sistema CutPlanner
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cutplanner import CutPlanner
from cutplanner.models import Material, Part, MaterialType, PartType
from cutplanner.utils import export_result, create_visualization


def create_sample_data():
    """Cria dados de exemplo para demonstração"""
    
    # Materiais de exemplo
    materials = [
        Material(
            id="barra_aco_6m",
            name="Barra de Aço 6m",
            material_type=MaterialType.BAR,
            length=6000,
            quantity=5,
            cost_per_unit=150.0
        ),
        Material(
            id="perfil_aluminio_4m",
            name="Perfil de Alumínio 4m",
            material_type=MaterialType.PROFILE,
            length=4000,
            quantity=3,
            cost_per_unit=120.0
        )
    ]
    
    # Peças de exemplo
    parts = [
        Part(
            id="viga_principal",
            name="Viga Principal",
            part_type=PartType.LINEAR,
            length=1200,
            quantity=10,
            priority=1
        ),
        Part(
            id="suporte_secundario",
            name="Suporte Secundário",
            part_type=PartType.LINEAR,
            length=800,
            quantity=15,
            priority=2
        ),
        Part(
            id="conector",
            name="Conector",
            part_type=PartType.LINEAR,
            length=600,
            quantity=20,
            priority=3
        )
    ]
    
    return materials, parts


def run_demo():
    """Executa demonstração do sistema"""
    
    print("🔧 CutPlanner - Demonstração do Sistema")
    print("=" * 60)
    
    # Criar dados de exemplo
    materials, parts = create_sample_data()
    
    # Criar planejador
    planner = CutPlanner(kerf_width=3.0)
    
    print(f"✓ Planejador configurado com espessura de corte: {planner.kerf_width}mm")
    print(f"✓ {len(materials)} materiais carregados")
    print(f"✓ {len(parts)} tipos de peças definidos")
    
    # Executar otimização
    print("\n🔄 Executando otimização...")
    result = planner.optimize_1d(
        materials=materials,
        parts=parts,
        algorithm="best_fit"
    )
    
    # Exibir resultados
    if result.success:
        print(f"\n✅ Otimização concluída com sucesso!")
        print(f"📊 Eficiência: {result.efficiency:.1f}%")
        print(f"🗑️  Desperdício: {result.total_waste:.1f}mm")
        print(f"📦 Materiais utilizados: {result.materials_used}")
        print(f"⚡ Tempo de processamento: {result.processing_time:.1f}ms")
        
        # Detalhes dos cortes
        print(f"\n📋 Resumo dos cortes:")
        for i, material_cut in enumerate(result.cuts, 1):
            print(f"  {i}. {material_cut.material_name}: {len(material_cut.cuts)} peças, eficiência {material_cut.efficiency:.1f}%")
        
        # Retalhos
        if result.leftovers:
            usable = [l for l in result.leftovers if l.usable]
            if usable:
                print(f"\n♻️  Retalhos utilizáveis: {len(usable)}")
                for leftover in usable[:3]:  # Mostrar apenas os 3 primeiros
                    print(f"     • {leftover.length:.1f}mm")
                if len(usable) > 3:
                    print(f"     • ... e mais {len(usable) - 3} retalhos")
        
        return result
    else:
        print(f"❌ Falha na otimização: {result.metadata.get('error', 'Erro desconhecido')}")
        return None


def run_api_server():
    """Inicia o servidor da API"""
    
    print("🚀 Iniciando servidor da API CutPlanner...")
    
    try:
        import uvicorn
        from api.main import app
        
        print("✓ Servidor iniciado em http://localhost:8000")
        print("✓ Documentação da API: http://localhost:8000/docs")
        print("✓ Interface web: http://localhost:8000/")
        print("\nPressione Ctrl+C para parar o servidor")
        
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Erro: {e}")
        print("Instale as dependências com: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")


def run_tests():
    """Executa os testes do sistema"""
    
    print("🧪 Executando testes do CutPlanner...")
    
    try:
        import unittest
        
        # Descobrir e executar testes
        loader = unittest.TestLoader()
        start_dir = Path(__file__).parent / 'tests'
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\n✅ Todos os testes passaram!")
            return True
        else:
            print(f"\n❌ {len(result.failures)} testes falharam")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao executar testes: {e}")
        return False


def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(
        description="CutPlanner - Sistema de Otimização de Cortes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run.py demo                    # Executa demonstração
  python run.py api                     # Inicia servidor da API
  python run.py test                    # Executa testes
  python run.py demo --export results   # Executa demo e exporta resultados
        """
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'api', 'test'],
        help='Comando a executar'
    )
    
    parser.add_argument(
        '--export',
        metavar='DIR',
        help='Diretório para exportar resultados'
    )
    
    parser.add_argument(
        '--visualization',
        action='store_true',
        help='Criar visualizações dos resultados'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'demo':
            result = run_demo()
            
            if result and args.export:
                print(f"\n📁 Exportando resultados para: {args.export}")
                export_result(result, args.export)
                
                if args.visualization:
                    print("🎨 Criando visualizações...")
                    create_visualization(result, args.export)
                
                print("✅ Exportação concluída!")
                
        elif args.command == 'api':
            run_api_server()
            
        elif args.command == 'test':
            success = run_tests()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 