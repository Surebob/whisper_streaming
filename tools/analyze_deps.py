import os
import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple
import logging
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImportVisitor(ast.NodeVisitor):
    """AST visitor for extracting imports."""
    def __init__(self, module_path: str):
        self.imports = set()
        self.module_path = module_path
        self.module_dir = os.path.dirname(module_path)
        
    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name)
            
    def visit_ImportFrom(self, node):
        if node.module is None:  # Handle "from . import x"
            if node.level > 0:  # Relative import
                parent = self.get_relative_module(node.level)
                for name in node.names:
                    self.imports.add(f"{parent}.{name.name}" if parent else name.name)
        else:
            if node.level > 0:  # Relative import with module
                parent = self.get_relative_module(node.level)
                module = f"{parent}.{node.module}" if parent else node.module
            else:
                module = node.module
            self.imports.add(module)
            for name in node.names:
                if name.name != '*':
                    self.imports.add(f"{module}.{name.name}")
                    
    def get_relative_module(self, level: int) -> str:
        """Convert relative import level to absolute module path."""
        parts = self.module_path.split(os.sep)
        if level > len(parts):
            return ""
        return ".".join(parts[:-level])

class DependencyAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.graph = nx.DiGraph()
        self.import_map: Dict[str, Set[str]] = {}
        self.module_files: Dict[str, str] = {}  # Maps module names to file paths
        
        # Define module categories and their colors
        self.categories = {
            'core': '#ADD8E6',  # Light blue
            'models': '#98FB98',  # Light green
            'utils': '#FFB6C1',  # Light pink
            'tests': '#F0E68C',  # Khaki
            'ui': '#DDA0DD',     # Plum
        }
        
    def get_module_category(self, module_name: str) -> Tuple[str, str]:
        """Determine the category and color for a module."""
        if 'test' in module_name:
            return 'tests', self.categories['tests']
        elif 'model' in module_name or 'vad' in module_name or 'whisper' in module_name:
            return 'models', self.categories['models']
        elif 'util' in module_name or 'helper' in module_name:
            return 'utils', self.categories['utils']
        elif 'ui' in module_name or 'display' in module_name:
            return 'ui', self.categories['ui']
        else:
            return 'core', self.categories['core']
    
    def module_name_from_path(self, file_path: str) -> str:
        """Convert file path to module name."""
        rel_path = os.path.relpath(file_path, self.root_dir)
        module_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
        return module_name
    
    def parse_file(self, file_path: str) -> Set[str]:
        """Parse a Python file and extract its imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)
            
            visitor = ImportVisitor(self.module_name_from_path(file_path))
            visitor.visit(tree)
            return visitor.imports
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return set()
    
    def find_python_files(self) -> List[str]:
        """Find all Python files in the project."""
        python_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)
        return python_files
    
    def resolve_import(self, import_name: str, current_module: str) -> str:
        """Resolve an import name to a module name."""
        parts = import_name.split('.')
        
        # Try direct module match
        if import_name in self.module_files:
            return import_name
            
        # Try as submodule
        for i in range(len(parts), 0, -1):
            potential_module = '.'.join(parts[:i])
            if potential_module in self.module_files:
                return potential_module
                
        # Try relative to current module
        current_parts = current_module.split('.')
        for i in range(len(current_parts)):
            potential_module = '.'.join(current_parts[:-(i+1)] + [import_name])
            if potential_module in self.module_files:
                return potential_module
                
        return import_name
    
    def build_graph(self):
        """Build the dependency graph by walking through Python files."""
        # First pass: collect all modules
        python_files = self.find_python_files()
        for file_path in python_files:
            module_name = self.module_name_from_path(file_path)
            self.module_files[module_name] = file_path
            
            # Add node with category information
            category, color = self.get_module_category(module_name)
            self.graph.add_node(module_name, 
                              category=category, 
                              color=color,
                              file_path=file_path)
        
        # Second pass: analyze imports
        for module_name, file_path in self.module_files.items():
            imports = self.parse_file(file_path)
            resolved_imports = set()
            
            for imp in imports:
                resolved = self.resolve_import(imp, module_name)
                if resolved in self.module_files:
                    resolved_imports.add(resolved)
            
            self.import_map[module_name] = resolved_imports
            
            # Add edges for dependencies
            for imp in resolved_imports:
                if imp in self.module_files:
                    self.graph.add_edge(module_name, imp)
    
    def find_cycles(self) -> List[List[str]]:
        """Find import cycles in the dependency graph."""
        return list(nx.simple_cycles(self.graph))
    
    def get_module_layers(self) -> Dict[str, int]:
        """Determine the layer of each module based on dependencies."""
        layers = {}
        remaining_nodes = set(self.graph.nodes())
        
        layer = 0
        while remaining_nodes:
            # Find nodes with no incoming edges from remaining nodes
            layer_nodes = {
                node for node in remaining_nodes
                if not any(pred in remaining_nodes 
                          for pred in self.graph.predecessors(node))
            }
            
            if not layer_nodes:  # Handle cycles by picking arbitrary node
                layer_nodes = {next(iter(remaining_nodes))}
            
            # Assign layer to nodes
            for node in layer_nodes:
                layers[node] = layer
            
            remaining_nodes -= layer_nodes
            layer += 1
        
        return layers
    
    def show_file_inventory(self):
        """Display a complete inventory of all Python files and their relationships."""
        logger.info("\n=== Complete File Inventory ===\n")
        
        # Group files by category
        files_by_category = {}
        for module_name, file_path in self.module_files.items():
            category = self.graph.nodes[module_name]['category']
            if category not in files_by_category:
                files_by_category[category] = []
            files_by_category[category].append((module_name, file_path))
        
        # Print files by category with their dependencies
        for category in sorted(files_by_category.keys()):
            logger.info(f"\n[{category.upper()}]")
            files = sorted(files_by_category[category])
            
            for module_name, file_path in files:
                rel_path = os.path.relpath(file_path, self.root_dir)
                logger.info(f"\n{rel_path}")
                
                # Show imports (dependencies)
                deps = list(self.graph.successors(module_name))
                if deps:
                    logger.info("  Imports:")
                    for dep in sorted(deps):
                        dep_path = os.path.relpath(self.module_files[dep], self.root_dir)
                        logger.info(f"    → {dep_path}")
                
                # Show imported by (reverse dependencies)
                imported_by = list(self.graph.predecessors(module_name))
                if imported_by:
                    logger.info("  Imported by:")
                    for imp in sorted(imported_by):
                        imp_path = os.path.relpath(self.module_files[imp], self.root_dir)
                        logger.info(f"    ← {imp_path}")
                        
                # Show file size
                size_bytes = os.path.getsize(file_path)
                size_kb = size_bytes / 1024
                logger.info(f"  Size: {size_kb:.1f} KB")
                
                # Show last modified time
                mod_time = os.path.getmtime(file_path)
                mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                logger.info(f"  Last modified: {mod_time_str}")
    
    def analyze_dependencies(self):
        """Analyze dependencies and print insights."""
        # Find cycles
        cycles = self.find_cycles()
        if cycles:
            logger.warning("\nFound import cycles:")
            for cycle in cycles:
                logger.warning(f"  {' -> '.join(cycle + [cycle[0]])}")
        
        # Calculate metrics
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Most imported modules
        most_imported = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nMost imported modules:")
        for module, count in most_imported:
            category = self.graph.nodes[module]['category']
            file_path = self.graph.nodes[module]['file_path']
            logger.info(f"  {module} ({category}): {count} imports")
            logger.info(f"    File: {file_path}")
        
        # Modules with most dependencies
        most_deps = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nModules with most dependencies:")
        for module, count in most_deps:
            category = self.graph.nodes[module]['category']
            file_path = self.graph.nodes[module]['file_path']
            logger.info(f"  {module} ({category}): {count} dependencies")
            logger.info(f"    File: {file_path}")
            # List actual dependencies
            deps = list(self.graph.successors(module))
            for dep in deps:
                logger.info(f"    → {dep}")
        
        # Analyze module categories
        categories = {}
        for node, data in self.graph.nodes(data=True):
            cat = data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info("\nModule distribution by category:")
        for cat, count in categories.items():
            logger.info(f"  {cat}: {count} modules")
            
        # Show complete file inventory
        self.show_file_inventory()
    
    def visualize(self, output_file: str = "dependency_graph.png"):
        """Visualize the dependency graph with improved hierarchy and relationships."""
        plt.figure(figsize=(20, 15))
        
        # Use hierarchical layout for better structure
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Get node colors from categories
        node_colors = [self.graph.nodes[node]['color'] for node in self.graph.nodes()]
        
        # Draw nodes with different sizes based on importance
        node_sizes = [
            3000 + 1000 * (self.graph.in_degree(node) + self.graph.out_degree(node))
            for node in self.graph.nodes()
        ]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes, 
                             alpha=0.7)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(self.graph, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             alpha=0.5,
                             connectionstyle="arc3,rad=0.2")
        
        # Add labels with better formatting
        labels = {
            node: f"{node.split('.')[-1]}\n({self.graph.nodes[node]['category']})"
            for node in self.graph.nodes()
        }
        nx.draw_networkx_labels(self.graph, pos, labels,
                              font_size=8,
                              font_weight='bold')
        
        # Add legend
        legend_elements = [
            Patch(facecolor=color, label=category.capitalize())
            for category, color in self.categories.items()
        ]
        plt.legend(handles=legend_elements,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))
        
        plt.title("Codebase Dependency Graph\nNode size indicates connectivity")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nDependency graph saved to {output_file}")

    def show_relative_paths(self):
        """Display all relative paths from project root, organized by category."""
        logger.info("\n=== Project Files (Relative Paths) ===\n")
        
        # Group files by category
        files_by_category = {}
        for module_name, file_path in self.module_files.items():
            category = self.graph.nodes[module_name]['category']
            if category not in files_by_category:
                files_by_category[category] = []
            rel_path = os.path.relpath(file_path, self.root_dir).replace('\\', '/')
            files_by_category[category].append(rel_path)
        
        # Print paths by category
        total_files = 0
        for category in sorted(files_by_category.keys()):
            files = sorted(files_by_category[category])
            logger.info(f"\n[{category.upper()}] ({len(files)} files)")
            for file_path in files:
                logger.info(f"  {file_path}")
            total_files += len(files)
        
        logger.info(f"\nTotal Python files: {total_files}")

def main():
    # Initialize analyzer with the project root
    analyzer = DependencyAnalyzer('.')
    
    # Build graph
    analyzer.build_graph()
    
    # Show relative paths
    analyzer.show_relative_paths()
    
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main() 