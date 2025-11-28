"""
AST CLI - Command-line interface for AST analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure backends are imported for registration
from ragix_core.ast_python import get_python_backend
try:
    from ragix_core.ast_java import get_java_backend, is_java_available
except ImportError:
    is_java_available = lambda: False

from ragix_core.ast_base import (
    ASTNode,
    Language,
    NodeType,
    format_ast_tree,
    get_ast_registry,
    count_nodes,
)
from ragix_core.dependencies import (
    DependencyGraph,
    DependencyType,
    build_dependency_graph,
)
from ragix_core.ast_query import (
    parse_query,
    execute_query,
    execute_query_on_ast,
)
from ragix_core.ast_viz import (
    VizConfig,
    LayoutDirection,
    ColorScheme,
    DotRenderer,
    MermaidRenderer,
    D3Renderer,
    HTMLRenderer,
    DSMRenderer,
    RadialExplorer,
    render_ast_tree,
)
from ragix_core.maven import (
    MavenParser,
    MavenProjectScanner,
    parse_pom,
    scan_maven_projects,
    find_dependency_conflicts,
)
try:
    from ragix_core.sonar import (
        SonarClient,
        SonarSeverity,
        SonarIssueType,
        get_project_report,
    )
    SONAR_AVAILABLE = True
except ImportError:
    SONAR_AVAILABLE = False

from ragix_core.code_metrics import (
    calculate_project_metrics,
    calculate_metrics_from_graph,
    PythonMetricsCalculator,
    ProjectMetrics,
)

# Advanced visualizations
try:
    from ragix_core.ast_viz_advanced import (
        TreemapConfig,
        TreemapMetric,
        TreemapRenderer,
        SunburstConfig,
        SunburstRenderer,
        ChordConfig,
        ChordRenderer,
    )
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False

# Report generation
try:
    from ragix_core.report_engine import (
        ReportEngine,
        ReportConfig,
        ReportData,
        ReportFormat,
        ReportType,
        ComplianceStandard,
        generate_executive_summary,
        generate_technical_audit,
        generate_compliance_report,
    )
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse a file and show the AST."""
    path = Path(args.file)

    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    registry = get_ast_registry()
    backend = registry.get_backend_for_file(path)

    if not backend:
        print(f"Error: No parser for file type: {path.suffix}", file=sys.stderr)
        return 1

    ast = backend.parse_file(path)

    if ast.metadata.get("parse_error"):
        print(f"Parse error: {ast.metadata.get('error')}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(ast.to_dict(), indent=2))
    elif args.symbols:
        symbols = backend.get_symbols(ast)
        print(f"Symbols in {path.name}:")
        print("=" * 60)
        for sym in symbols:
            vis = f"[{sym.visibility.value}]" if sym.visibility.value != "unknown" else ""
            print(f"  {sym.node_type.value:12} {vis:10} {sym.qualified_name}")
        print(f"\nTotal: {len(symbols)} symbols")
    else:
        print(f"AST for {path.name}:")
        print("=" * 60)
        print(format_ast_tree(ast, max_depth=args.depth))

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a directory and build dependency graph."""
    path = Path(args.directory)

    if not path.exists():
        print(f"Error: Directory not found: {path}", file=sys.stderr)
        return 1

    # Parse language patterns
    patterns = []
    if args.lang:
        lang_map = {
            "py": "*.py",
            "python": "*.py",
            "java": "*.java",
            "js": "*.js",
            "javascript": "*.js",
            "ts": "*.ts",
            "typescript": "*.ts",
        }
        for lang in args.lang.split(","):
            lang = lang.strip().lower()
            if lang in lang_map:
                patterns.append(lang_map[lang])
            elif lang.startswith("*."):
                patterns.append(lang)
    else:
        patterns = ["*.py", "*.java"]

    print(f"Scanning {path} for {', '.join(patterns)}...")

    graph = DependencyGraph()
    stats = graph.add_directory(path, patterns, recursive=not args.no_recursive)

    print(f"Parsed: {stats['parsed']} files, Failed: {stats['failed']} files")
    print(f"Found: {len(graph.get_symbols())} symbols, {len(graph)} dependencies")

    if args.stats:
        dep_stats = graph.get_stats()
        print("\nDependency breakdown:")
        for dt, count in sorted(dep_stats.by_type.items(), key=lambda x: -x[1]):
            print(f"  {dt.value:20} {count}")

        if dep_stats.cycles:
            print(f"\nCircular dependencies: {len(dep_stats.cycles)}")
            for i, cycle in enumerate(dep_stats.cycles[:5], 1):
                print(f"  {i}. {' -> '.join(cycle)}")
            if len(dep_stats.cycles) > 5:
                print(f"  ... and {len(dep_stats.cycles) - 5} more")

    return 0


def cmd_deps(args: argparse.Namespace) -> int:
    """Show dependencies for a symbol."""
    path = Path(args.path)

    if path.is_file():
        graph = DependencyGraph()
        graph.add_file(path)
    elif path.is_dir():
        graph = build_dependency_graph([path])
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    symbol = args.symbol
    symbols = graph.find_symbols(symbol)

    if not symbols:
        print(f"Symbol not found: {symbol}")
        print("\nAvailable symbols (sample):")
        for s in list(graph.get_symbols())[:10]:
            print(f"  {s.qualified_name}")
        return 1

    for sym in symbols:
        print(f"\n{sym.node_type.value}: {sym.qualified_name}")
        print(f"  Location: {sym.location}")
        print()

        # Outgoing dependencies
        deps = graph.get_dependencies(sym.qualified_name)
        if deps:
            print("  Dependencies (uses):")
            for dep in deps[:20]:
                print(f"    {dep.dep_type.value:15} -> {dep.target}")
            if len(deps) > 20:
                print(f"    ... and {len(deps) - 20} more")

        # Incoming dependencies
        dependents = graph.get_dependents(sym.qualified_name)
        if dependents:
            print("\n  Dependents (used by):")
            for dep in dependents[:20]:
                print(f"    {dep.dep_type.value:15} <- {dep.source}")
            if len(dependents) > 20:
                print(f"    ... and {len(dependents) - 20} more")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search for symbols using query language."""
    path = Path(args.path)

    if path.is_file():
        graph = DependencyGraph()
        graph.add_file(path)
    elif path.is_dir():
        patterns = ["*.py", "*.java"] if not args.lang else [f"*.{args.lang}"]
        graph = DependencyGraph()
        graph.add_directory(path, patterns)
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    query = parse_query(args.query)
    matches = execute_query(query, graph)

    print(f'Search: "{args.query}"')
    print(f"Found: {len(matches)} matches")
    print("=" * 60)

    for match in matches[:args.limit]:
        if match.symbol:
            sym = match.symbol
            vis = f"[{sym.visibility.value}]" if sym.visibility.value != "unknown" else ""
            print(f"{sym.node_type.value:12} {vis:10} {sym.qualified_name}")
            if args.verbose:
                print(f"             Location: {sym.location}")

    if len(matches) > args.limit:
        print(f"\n... and {len(matches) - args.limit} more")

    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Generate dependency graph visualization."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    if path.is_file():
        graph = DependencyGraph()
        graph.add_file(path)
    else:
        graph = build_dependency_graph([path])

    # Build visualization config
    config = VizConfig()

    # Direction
    if hasattr(args, 'direction') and args.direction:
        dir_map = {
            "lr": LayoutDirection.LEFT_RIGHT,
            "tb": LayoutDirection.TOP_BOTTOM,
            "rl": LayoutDirection.RIGHT_LEFT,
            "bt": LayoutDirection.BOTTOM_TOP,
        }
        config.direction = dir_map.get(args.direction.lower(), LayoutDirection.LEFT_RIGHT)

    # Color scheme
    if hasattr(args, 'colors') and args.colors:
        color_map = {
            "default": ColorScheme.DEFAULT,
            "pastel": ColorScheme.PASTEL,
            "dark": ColorScheme.DARK,
            "mono": ColorScheme.MONOCHROME,
        }
        config.color_scheme = color_map.get(args.colors.lower(), ColorScheme.DEFAULT)

    # Cluster by file
    if hasattr(args, 'cluster') and args.cluster:
        config.cluster_by_file = True

    # Filter dependency types
    if args.types:
        dep_types = []
        for t in args.types.split(","):
            try:
                dep_types.append(DependencyType(t.strip()))
            except ValueError:
                print(f"Warning: Unknown dependency type: {t}", file=sys.stderr)
        config.filter_deps = dep_types

    # Generate output
    fmt = args.format.lower()

    if fmt == "dot":
        renderer = DotRenderer(config)
        output = renderer.render(graph)
    elif fmt == "mermaid":
        renderer = MermaidRenderer(config)
        output = renderer.render(graph)
    elif fmt == "json":
        output = json.dumps(graph.export_json(), indent=2)
    elif fmt == "d3":
        renderer = D3Renderer(config)
        output = renderer.render(graph)
    elif fmt == "html":
        renderer = HTMLRenderer(config)
        title = path.name if path.is_file() else path.stem
        output = renderer.render(graph, title=f"Dependencies: {title}")
    else:
        print(f"Error: Unknown format: {fmt}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(output)
        print(f"Graph written to {args.output}")
    else:
        print(output)

    return 0


def cmd_cycles(args: argparse.Namespace) -> int:
    """Detect circular dependencies."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    graph = build_dependency_graph([path])
    cycles = graph.detect_cycles()

    if not cycles:
        print("No circular dependencies detected")
        return 0

    print(f"Circular Dependencies: {len(cycles)}")
    print("=" * 60)

    for i, cycle in enumerate(cycles, 1):
        print(f"\n{i}. Cycle of {len(cycle) - 1} symbols:")
        for j, sym in enumerate(cycle):
            arrow = " -> " if j < len(cycle) - 1 else ""
            print(f"   {sym}{arrow}", end="")
        print()

    return 1 if cycles else 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Generate dependency structure matrix (DSM)."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])

    # Create DSM renderer
    config = VizConfig()
    renderer = DSMRenderer(config)

    # Get aggregation level
    level = args.level if hasattr(args, 'level') and args.level else "class"

    if args.csv:
        # Export as CSV
        csv_content = renderer.to_csv(graph, level=level)
        if hasattr(args, 'output') and args.output:
            output_path = Path(args.output)
            output_path.write_text(csv_content)
            print(f"CSV exported to: {output_path}")
        else:
            print(csv_content)
    elif args.json:
        # Export as JSON
        data = renderer.to_matrix(graph, level=level)
        print(json.dumps(data, indent=2))
    else:
        # Generate HTML
        html_content = renderer.render_html(
            graph,
            title=f"Dependency Matrix - {path.name}",
            level=level
        )

        if hasattr(args, 'output') and args.output:
            output_path = Path(args.output)
            output_path.write_text(html_content)
            print(f"Matrix visualization saved to: {output_path}")
            print(f"Open in browser: xdg-open {output_path}")
        else:
            # Write to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False
            ) as f:
                f.write(html_content)
                print(f"Matrix visualization saved to: {f.name}")
                print(f"Open in browser: xdg-open {f.name}")

    return 0


def cmd_radial(args: argparse.Namespace) -> int:
    """Generate radial/ego-centric dependency explorer."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])
    symbols = {s.qualified_name: s for s in graph.get_symbols()}

    # Determine focal node
    focal_node = args.focal if hasattr(args, 'focal') and args.focal else None

    if not focal_node:
        # Auto-select: find the class with most connections (highest degree)
        print("No focal node specified, selecting class with highest connectivity...")
        deps = graph.get_all_dependencies()

        # Count connections per node
        degree = {}
        for dep in deps:
            degree[dep.source] = degree.get(dep.source, 0) + 1
            degree[dep.target] = degree.get(dep.target, 0) + 1

        if not degree:
            print("Error: No dependencies found in the project", file=sys.stderr)
            return 1

        # Filter to prefer actual project classes (exclude standard library, annotations)
        excluded_prefixes = (
            'java.lang.', 'java.util.', 'java.io.', 'java.sql.', 'java.time.',
            'org.springframework.', 'javax.', 'lombok.', 'org.joda.',
            'org.slf4j.', 'com.fasterxml.', 'org.apache.', 'com.google.',
            'org.junit.', 'org.mockito.', 'org.hamcrest.', 'org.assertj.',
        )
        class_degree = {
            k: v for k, v in degree.items()
            if k in symbols and '.' in k and not k.startswith(excluded_prefixes)
        }

        if not class_degree:
            # Fallback: try any project classes
            class_degree = {k: v for k, v in degree.items() if k in symbols and '.' in k}

        if not class_degree:
            # Final fallback to all nodes
            class_degree = degree

        # Sort by degree
        focal_node = max(class_degree.keys(), key=lambda k: class_degree[k])
        print(f"Selected: {focal_node} ({class_degree[focal_node]} connections)")

    # Validate focal node exists (only if user specified it)
    if hasattr(args, 'focal') and args.focal and focal_node not in symbols:
        # Try partial match
        matches = [s for s in symbols.keys() if focal_node in s]
        if matches:
            if len(matches) == 1:
                focal_node = matches[0]
                print(f"Matched: {focal_node}")
            else:
                print(f"Error: Ambiguous focal node '{args.focal}'. Matches:")
                for m in matches[:10]:
                    print(f"  - {m}")
                if len(matches) > 10:
                    print(f"  ... and {len(matches) - 10} more")
                return 1
        else:
            print(f"Error: Focal node not found: {args.focal}", file=sys.stderr)
            return 1

    # Get max levels
    max_levels = args.levels if hasattr(args, 'levels') and args.levels else 3

    # Create radial explorer
    config = VizConfig()
    explorer = RadialExplorer(config)

    # Generate HTML
    html_content = explorer.render_html(
        graph,
        focal_node=focal_node,
        title=f"Radial Explorer - {focal_node.split('.')[-1]}",
        max_levels=max_levels
    )

    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.write_text(html_content)
        print(f"Radial visualization saved to: {output_path}")
        print(f"Open in browser: xdg-open {output_path}")
    else:
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False
        ) as f:
            f.write(html_content)
            print(f"Radial visualization saved to: {f.name}")
            print(f"Open in browser: xdg-open {f.name}")

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Show professional code metrics."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    # Build dependency graph for coupling analysis
    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])
    stats = graph.get_stats()

    # Calculate code metrics from graph
    code_metrics = calculate_metrics_from_graph(graph)

    print()
    print("=" * 70)
    print("  PROFESSIONAL CODE METRICS REPORT")
    print("=" * 70)

    # Overview
    summary = code_metrics.summary()
    print(f"\n{'PROJECT OVERVIEW':^70}")
    print("-" * 70)
    print(f"  Files analyzed:       {summary['files']:>10}")
    print(f"  Total lines:          {summary['lines']['total']:>10}")
    print(f"  Code lines:           {summary['lines']['code']:>10}")
    print(f"  Comment lines:        {summary['lines']['comments']:>10}")
    print(f"  Classes:              {summary['classes']:>10}")
    print(f"  Functions/Methods:    {summary['functions']:>10}")

    # Complexity metrics
    print(f"\n{'COMPLEXITY METRICS':^70}")
    print("-" * 70)
    print(f"  Total Cyclomatic Complexity: {summary['complexity']['total']:>10}")
    print(f"  Avg Complexity per Method:   {summary['complexity']['avg_per_method']:>10.2f}")

    # Technical debt
    print(f"\n{'TECHNICAL DEBT ESTIMATION':^70}")
    print("-" * 70)
    debt_hours = summary['technical_debt']['hours']
    debt_days = summary['technical_debt']['days']
    print(f"  Estimated Effort:     {debt_hours:>10.1f} hours ({debt_days:.1f} person-days)")

    # Maintainability
    mi = summary['maintainability_index']
    mi_rating = "Excellent" if mi >= 80 else "Good" if mi >= 60 else "Moderate" if mi >= 40 else "Poor"
    print(f"  Maintainability Index: {mi:>9.1f} ({mi_rating})")

    # Hotspots
    hotspots = code_metrics.get_hotspots(10)
    if hotspots:
        print(f"\n{'COMPLEXITY HOTSPOTS (Top 10)':^70}")
        print("-" * 70)
        for name, cc in hotspots:
            if cc > 1:
                indicator = "⚠️" if cc > 20 else "⚡" if cc > 10 else " "
                print(f"  {indicator} CC={cc:>3}  {name[:60]}")

    # Coupling Metrics
    print(f"\n{'COUPLING METRICS':^70}")
    print("-" * 70)
    print(f"  Total Symbols:        {len(graph.get_symbols()):>10}")
    print(f"  Total Dependencies:   {stats.total_dependencies:>10}")

    # Top afferent (most depended upon)
    print(f"\n  Most Used (Afferent Coupling - Top 10):")
    afferent = sorted(stats.afferent_coupling.items(), key=lambda x: -x[1])
    for sym, count in afferent[:10]:
        if count > 0:
            print(f"    {count:>4} <- {sym[:55]}")

    # Top efferent (depends on most)
    print(f"\n  Most Dependencies (Efferent Coupling - Top 10):")
    efferent = sorted(stats.efferent_coupling.items(), key=lambda x: -x[1])
    for sym, count in efferent[:10]:
        if count > 0:
            print(f"    {count:>4} -> {sym[:55]}")

    # Instability Index
    print(f"\n  Instability Index (0=stable, 1=unstable):")
    instability = []
    for sym in stats.afferent_coupling:
        ca = stats.afferent_coupling.get(sym, 0)
        ce = stats.efferent_coupling.get(sym, 0)
        if ca + ce > 0:
            i = ce / (ca + ce)
            instability.append((sym, i, ca, ce))

    instability.sort(key=lambda x: -x[1])
    for sym, i, ca, ce in instability[:10]:
        if ca + ce >= 3:
            print(f"    {i:.2f}  {sym[:50]} (Ca={ca}, Ce={ce})")

    # Circular dependencies
    if stats.cycles:
        print(f"\n{'⚠️  CIRCULAR DEPENDENCIES DETECTED':^70}")
        print("-" * 70)
        print(f"  Found {len(stats.cycles)} circular dependency chain(s)")
        for i, cycle in enumerate(stats.cycles[:5], 1):
            print(f"    {i}. {' -> '.join(c[:20] for c in cycle[:4])}")

    # JSON output
    if hasattr(args, 'json') and args.json:
        output = {
            "summary": summary,
            "coupling": {
                "symbols": len(graph.get_symbols()),
                "dependencies": stats.total_dependencies,
                "cycles": len(stats.cycles),
            },
            "hotspots": [{"name": n, "complexity": c} for n, c in hotspots],
        }
        print("\n" + json.dumps(output, indent=2))

    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about supported languages."""
    registry = get_ast_registry()

    print("Supported Languages")
    print("=" * 40)

    for lang in registry.list_languages():
        backend = registry.get_backend(lang)
        if backend:
            exts = ", ".join(backend.file_extensions)
            print(f"  {lang.value:12} {exts}")

    print()
    print("Query Syntax Help")
    print("=" * 40)
    print("  type:pattern        Match node type (class, method, function)")
    print("  name:pattern        Match by name")
    print("  extends:pattern     Match classes extending")
    print("  implements:pattern  Match classes implementing")
    print("  calls:pattern       Match methods that call")
    print("  @annotation         Match by decorator/annotation")
    print("  visibility:public   Match by visibility")
    print("  !predicate          Negate a predicate")
    print()
    print("Examples:")
    print('  ragix-ast search . "type:class name:*Service"')
    print('  ragix-ast search . "extends:Base*"')
    print('  ragix-ast search . "@Override"')

    return 0


def cmd_maven(args: argparse.Namespace) -> int:
    """Analyze Maven projects."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    # Check if it's a pom.xml file or directory
    if path.is_file() and path.name == "pom.xml":
        projects = [parse_pom(path)]
    elif path.is_dir():
        print(f"Scanning {path} for Maven projects...")
        projects = scan_maven_projects(path)
    else:
        print(f"Error: Expected pom.xml or directory: {path}", file=sys.stderr)
        return 1

    if not projects:
        print("No Maven projects found")
        return 0

    print(f"Found {len(projects)} Maven project(s)")
    print("=" * 60)

    for project in projects:
        print(f"\nProject: {project.coordinate.gav}")
        if project.name:
            print(f"  Name: {project.name}")
        print(f"  Packaging: {project.packaging}")
        if project.pom_path:
            print(f"  Path: {project.pom_path}")

        if project.parent:
            print(f"  Parent: {project.parent.gav}")

        if project.modules:
            print(f"  Modules: {len(project.modules)}")
            for mod in project.modules[:5]:
                print(f"    - {mod}")
            if len(project.modules) > 5:
                print(f"    ... and {len(project.modules) - 5} more")

        if project.dependencies:
            compile_deps = project.get_compile_dependencies()
            test_deps = project.get_test_dependencies()
            print(f"  Dependencies: {len(project.dependencies)} total")
            print(f"    Compile: {len(compile_deps)}")
            print(f"    Test: {len(test_deps)}")

            if args.verbose:
                print("\n  Compile dependencies:")
                for dep in compile_deps[:10]:
                    opt = " (optional)" if dep.optional else ""
                    print(f"    - {dep.coordinate.gav}{opt}")
                if len(compile_deps) > 10:
                    print(f"    ... and {len(compile_deps) - 10} more")

    # Check for conflicts if multiple projects
    if len(projects) > 1 and args.conflicts:
        conflicts = find_dependency_conflicts(projects)
        if conflicts:
            print(f"\nDependency Conflicts: {len(conflicts)}")
            print("-" * 40)
            for conflict in conflicts[:10]:
                print(f"\n  {conflict['artifact']}:")
                for version, users in conflict['versions'].items():
                    print(f"    {version}: {len(users)} project(s)")
            if len(conflicts) > 10:
                print(f"\n  ... and {len(conflicts) - 10} more conflicts")
        else:
            print("\nNo dependency conflicts found")

    # JSON output
    if args.json:
        output = {
            "projects": [
                {
                    "gav": p.coordinate.gav,
                    "name": p.name,
                    "packaging": p.packaging,
                    "parent": p.parent.gav if p.parent else None,
                    "modules": p.modules,
                    "dependencies": [
                        {
                            "gav": d.coordinate.gav,
                            "scope": d.scope.value,
                            "optional": d.optional,
                        }
                        for d in p.dependencies
                    ],
                }
                for p in projects
            ],
        }
        print("\n" + json.dumps(output, indent=2))

    return 0


def cmd_sonar(args: argparse.Namespace) -> int:
    """Query SonarQube/SonarCloud for project metrics."""
    if not SONAR_AVAILABLE:
        print("Error: Sonar integration requires 'requests' library", file=sys.stderr)
        print("Install with: pip install requests", file=sys.stderr)
        return 1

    # Get connection params
    base_url = args.url or os.environ.get("SONAR_URL", "https://sonarcloud.io")
    token = args.token or os.environ.get("SONAR_TOKEN")
    org = args.organization or os.environ.get("SONAR_ORGANIZATION")

    if not token:
        print("Warning: No SONAR_TOKEN provided. API may have limited access.")

    try:
        client = SonarClient(base_url=base_url, token=token, organization=org)
    except Exception as e:
        print(f"Error connecting to Sonar: {e}", file=sys.stderr)
        return 1

    project_key = args.project

    print(f"Fetching Sonar data for: {project_key}")
    print(f"Server: {base_url}")
    print("=" * 60)

    try:
        report = get_project_report(client, project_key)
    except Exception as e:
        print(f"Error fetching project: {e}", file=sys.stderr)
        return 1

    project = report.project

    # Quality gate
    gate_status = {
        "OK": "PASSED",
        "WARN": "WARNING",
        "ERROR": "FAILED",
        "NONE": "N/A",
    }
    print(f"\nQuality Gate: {gate_status.get(project.quality_gate.value, 'N/A')}")

    # Metrics
    print(f"\nMetrics:")
    print(f"  Bugs:            {project.bugs}")
    print(f"  Vulnerabilities: {project.vulnerabilities}")
    print(f"  Code Smells:     {project.code_smells}")

    if project.coverage is not None:
        print(f"  Coverage:        {project.coverage:.1f}%")
    if project.duplicated_lines_density is not None:
        print(f"  Duplication:     {project.duplicated_lines_density:.1f}%")

    # Issues summary
    if report.issues:
        print(f"\nOpen Issues: {len(report.issues)}")

        # Count by severity
        by_severity = {}
        for issue in report.issues:
            sev = issue.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        for sev in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]:
            if sev in by_severity:
                print(f"  {sev:10} {by_severity[sev]}")

        # Show top issues if verbose
        if args.verbose:
            print("\nTop Issues:")
            for issue in report.issues[:10]:
                loc = f":{issue.line}" if issue.line else ""
                print(f"  [{issue.severity.value}] {issue.rule}")
                print(f"    {issue.message}")
                print(f"    {issue.file_path}{loc}")

    # Hotspots
    if report.hotspots:
        print(f"\nSecurity Hotspots: {len(report.hotspots)}")

    # Markdown output
    if args.markdown:
        print("\n" + "=" * 60)
        print(report.to_markdown())

    # JSON output
    if args.json:
        print("\n" + json.dumps(report.summary(), indent=2))

    return 0


def cmd_treemap(args: argparse.Namespace) -> int:
    """Generate treemap visualization."""
    if not ADVANCED_VIZ_AVAILABLE:
        print("Error: Advanced visualizations require 'jinja2'", file=sys.stderr)
        print("Install with: pip install jinja2", file=sys.stderr)
        return 1

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])

    # Configure treemap
    metric_map = {
        "loc": TreemapMetric.LOC,
        "complexity": TreemapMetric.COMPLEXITY,
        "count": TreemapMetric.COUNT,
        "debt": TreemapMetric.DEBT,
    }
    config = TreemapConfig(
        metric=metric_map.get(args.metric, TreemapMetric.LOC),
        title=args.title or f"Treemap - {path.name}",
        max_depth=args.depth,
    )

    renderer = TreemapRenderer(config)
    html_content = renderer.render(graph)

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.write_text(html_content)
        print(f"Treemap saved to: {output_path}")
        print(f"Open in browser: xdg-open {output_path}")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            print(f"Treemap saved to: {f.name}")
            print(f"Open in browser: xdg-open {f.name}")

    return 0


def cmd_sunburst(args: argparse.Namespace) -> int:
    """Generate sunburst visualization."""
    if not ADVANCED_VIZ_AVAILABLE:
        print("Error: Advanced visualizations require 'jinja2'", file=sys.stderr)
        print("Install with: pip install jinja2", file=sys.stderr)
        return 1

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])

    config = SunburstConfig(
        title=args.title or f"Sunburst - {path.name}",
        max_depth=args.depth,
    )

    renderer = SunburstRenderer(config)
    html_content = renderer.render(graph)

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.write_text(html_content)
        print(f"Sunburst saved to: {output_path}")
        print(f"Open in browser: xdg-open {output_path}")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            print(f"Sunburst saved to: {f.name}")
            print(f"Open in browser: xdg-open {f.name}")

    return 0


def cmd_chord(args: argparse.Namespace) -> int:
    """Generate chord diagram visualization."""
    if not ADVANCED_VIZ_AVAILABLE:
        print("Error: Advanced visualizations require 'jinja2'", file=sys.stderr)
        print("Install with: pip install jinja2", file=sys.stderr)
        return 1

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])

    config = ChordConfig(
        title=args.title or f"Dependencies - {path.name}",
        group_by=args.group_by,
        min_connections=args.min_connections,
    )

    renderer = ChordRenderer(config)
    html_content = renderer.render(graph)

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.write_text(html_content)
        print(f"Chord diagram saved to: {output_path}")
        print(f"Open in browser: xdg-open {output_path}")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            print(f"Chord diagram saved to: {f.name}")
            print(f"Open in browser: xdg-open {f.name}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate analysis reports."""
    if not REPORTS_AVAILABLE:
        print("Error: Report generation requires 'jinja2'", file=sys.stderr)
        print("Install with: pip install jinja2", file=sys.stderr)
        return 1

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    print(f"Analyzing {path}...")
    graph = build_dependency_graph([path])
    code_metrics = calculate_metrics_from_graph(graph)

    # Determine output format
    fmt = ReportFormat.PDF if args.pdf else ReportFormat.HTML

    # Check PDF availability
    if fmt == ReportFormat.PDF:
        try:
            from weasyprint import HTML
        except ImportError:
            print("Warning: weasyprint not available, falling back to HTML", file=sys.stderr)
            fmt = ReportFormat.HTML

    # Determine output path
    output_path = Path(args.output) if args.output else None

    # Generate appropriate report
    report_type = args.type.lower()
    project_name = args.project or path.name

    if report_type == "executive":
        print("Generating Executive Summary...")
        html_content = generate_executive_summary(
            metrics=code_metrics,
            graph=graph,
            project_name=project_name,
            output_path=output_path,
            format=fmt
        )
    elif report_type == "technical":
        print("Generating Technical Audit Report...")
        html_content = generate_technical_audit(
            metrics=code_metrics,
            graph=graph,
            project_name=project_name,
            output_path=output_path,
            format=fmt
        )
    elif report_type == "compliance":
        print("Generating Compliance Report...")
        standard_map = {
            "sonarqube": ComplianceStandard.SONARQUBE,
            "owasp": ComplianceStandard.OWASP,
            "iso25010": ComplianceStandard.ISO_25010,
        }
        standard = standard_map.get(args.standard, ComplianceStandard.SONARQUBE)
        html_content = generate_compliance_report(
            metrics=code_metrics,
            graph=graph,
            project_name=project_name,
            standard=standard,
            output_path=output_path,
            format=fmt
        )
    else:
        print(f"Error: Unknown report type: {report_type}", file=sys.stderr)
        print("Available types: executive, technical, compliance", file=sys.stderr)
        return 1

    # Handle output
    if output_path:
        if fmt == ReportFormat.HTML and not output_path.suffix:
            output_path = output_path.with_suffix('.html')
        if not output_path.exists() or fmt == ReportFormat.HTML:
            # HTML not yet written (PDF was written by generator)
            if fmt == ReportFormat.HTML:
                output_path.write_text(html_content)
        print(f"Report saved to: {output_path}")
        if fmt == ReportFormat.HTML:
            print(f"Open in browser: xdg-open {output_path}")
    else:
        import tempfile
        suffix = '.html'
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(html_content)
            print(f"Report saved to: {f.name}")
            print(f"Open in browser: xdg-open {f.name}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="ragix-ast",
        description="AST analysis and dependency tracking",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.11.1",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # parse command
    parse_parser = subparsers.add_parser("parse", help="Parse a file and show AST")
    parse_parser.add_argument("file", help="File to parse")
    parse_parser.add_argument("--json", action="store_true", help="Output as JSON")
    parse_parser.add_argument("--symbols", "-s", action="store_true", help="Show symbols only")
    parse_parser.add_argument("--depth", "-d", type=int, default=10, help="Max tree depth")
    parse_parser.set_defaults(func=cmd_parse)

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan directory for code")
    scan_parser.add_argument("directory", help="Directory to scan")
    scan_parser.add_argument("--lang", "-l", help="Languages (py,java,js)")
    scan_parser.add_argument("--no-recursive", action="store_true", help="Don't recurse")
    scan_parser.add_argument("--stats", "-s", action="store_true", help="Show statistics")
    scan_parser.set_defaults(func=cmd_scan)

    # deps command
    deps_parser = subparsers.add_parser("deps", help="Show dependencies")
    deps_parser.add_argument("path", help="File or directory")
    deps_parser.add_argument("symbol", help="Symbol name (supports wildcards)")
    deps_parser.set_defaults(func=cmd_deps)

    # search command
    search_parser = subparsers.add_parser("search", help="Search for symbols")
    search_parser.add_argument("path", help="File or directory")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("--lang", "-l", help="Language filter")
    search_parser.add_argument("--limit", "-n", type=int, default=50, help="Max results")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    search_parser.set_defaults(func=cmd_search)

    # graph command
    graph_parser = subparsers.add_parser("graph", help="Generate dependency graph")
    graph_parser.add_argument("path", help="File or directory")
    graph_parser.add_argument("--format", "-f", default="dot",
                              help="Output format (dot, mermaid, json, d3, html)")
    graph_parser.add_argument("--output", "-o", help="Output file")
    graph_parser.add_argument("--types", "-t", help="Dependency types to include")
    graph_parser.add_argument("--direction", "-d", default="lr",
                              help="Layout direction (lr, tb, rl, bt)")
    graph_parser.add_argument("--colors", "-c", default="default",
                              help="Color scheme (default, pastel, dark, mono)")
    graph_parser.add_argument("--cluster", action="store_true",
                              help="Cluster nodes by file")
    graph_parser.set_defaults(func=cmd_graph)

    # cycles command
    cycles_parser = subparsers.add_parser("cycles", help="Detect circular dependencies")
    cycles_parser.add_argument("path", help="File or directory")
    cycles_parser.set_defaults(func=cmd_cycles)

    # matrix command (DSM)
    matrix_parser = subparsers.add_parser("matrix", help="Generate dependency structure matrix")
    matrix_parser.add_argument("path", help="File or directory")
    matrix_parser.add_argument("--output", "-o", help="Output file")
    matrix_parser.add_argument("--level", "-l", default="class",
                               help="Aggregation level (class, package)")
    matrix_parser.add_argument("--csv", action="store_true", help="Output as CSV")
    matrix_parser.add_argument("--json", action="store_true", help="Output as JSON")
    matrix_parser.set_defaults(func=cmd_matrix)

    # radial command (ego-centric explorer)
    radial_parser = subparsers.add_parser("radial", help="Generate radial dependency explorer")
    radial_parser.add_argument("path", help="File or directory")
    radial_parser.add_argument("--focal", "-f", help="Focal node (class name or qualified name)")
    radial_parser.add_argument("--levels", "-l", type=int, default=3,
                               help="Maximum depth levels (default: 3)")
    radial_parser.add_argument("--output", "-o", help="Output file")
    radial_parser.set_defaults(func=cmd_radial)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Show professional code metrics")
    metrics_parser.add_argument("path", help="File or directory")
    metrics_parser.add_argument("--json", action="store_true", help="Output as JSON")
    metrics_parser.set_defaults(func=cmd_metrics)

    # info command
    info_parser = subparsers.add_parser("info", help="Show supported languages")
    info_parser.set_defaults(func=cmd_info)

    # maven command
    maven_parser = subparsers.add_parser("maven", help="Analyze Maven projects")
    maven_parser.add_argument("path", help="pom.xml file or directory to scan")
    maven_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Show detailed dependency list")
    maven_parser.add_argument("--conflicts", "-c", action="store_true",
                              help="Check for dependency version conflicts")
    maven_parser.add_argument("--json", action="store_true",
                              help="Output as JSON")
    maven_parser.set_defaults(func=cmd_maven)

    # sonar command
    sonar_parser = subparsers.add_parser("sonar", help="Query SonarQube/SonarCloud")
    sonar_parser.add_argument("project", help="Sonar project key")
    sonar_parser.add_argument("--url", "-u", help="Sonar server URL (or SONAR_URL env)")
    sonar_parser.add_argument("--token", "-t", help="API token (or SONAR_TOKEN env)")
    sonar_parser.add_argument("--organization", "-o",
                              help="Organization key for SonarCloud (or SONAR_ORGANIZATION env)")
    sonar_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Show detailed issue list")
    sonar_parser.add_argument("--markdown", "-m", action="store_true",
                              help="Output Markdown report")
    sonar_parser.add_argument("--json", action="store_true",
                              help="Output as JSON")
    sonar_parser.set_defaults(func=cmd_sonar)

    # treemap command (advanced visualization)
    treemap_parser = subparsers.add_parser("treemap", help="Generate treemap visualization")
    treemap_parser.add_argument("path", help="File or directory")
    treemap_parser.add_argument("--output", "-o", help="Output file")
    treemap_parser.add_argument("--metric", "-m", default="loc",
                                help="Size metric (loc, complexity, count, debt)")
    treemap_parser.add_argument("--depth", "-d", type=int, default=4,
                                help="Maximum depth (default: 4)")
    treemap_parser.add_argument("--title", "-t", help="Custom title")
    treemap_parser.set_defaults(func=cmd_treemap)

    # sunburst command (advanced visualization)
    sunburst_parser = subparsers.add_parser("sunburst", help="Generate sunburst diagram")
    sunburst_parser.add_argument("path", help="File or directory")
    sunburst_parser.add_argument("--output", "-o", help="Output file")
    sunburst_parser.add_argument("--depth", "-d", type=int, default=5,
                                 help="Maximum depth (default: 5)")
    sunburst_parser.add_argument("--title", "-t", help="Custom title")
    sunburst_parser.set_defaults(func=cmd_sunburst)

    # chord command (advanced visualization)
    chord_parser = subparsers.add_parser("chord", help="Generate chord diagram")
    chord_parser.add_argument("path", help="File or directory")
    chord_parser.add_argument("--output", "-o", help="Output file")
    chord_parser.add_argument("--group-by", "-g", default="package",
                              help="Grouping (package, file)")
    chord_parser.add_argument("--min-connections", "-m", type=int, default=1,
                              help="Minimum connections to show")
    chord_parser.add_argument("--title", "-t", help="Custom title")
    chord_parser.set_defaults(func=cmd_chord)

    # report command (professional reports)
    report_parser = subparsers.add_parser("report", help="Generate analysis reports")
    report_parser.add_argument("path", help="File or directory")
    report_parser.add_argument("--type", "-t", default="executive",
                               help="Report type (executive, technical, compliance)")
    report_parser.add_argument("--output", "-o", help="Output file")
    report_parser.add_argument("--project", "-p", help="Project name for report")
    report_parser.add_argument("--pdf", action="store_true",
                               help="Generate PDF (requires weasyprint)")
    report_parser.add_argument("--standard", "-s", default="sonarqube",
                               help="Compliance standard (sonarqube, owasp, iso25010)")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
