"""
Report Engine - Generate Professional PDF/HTML Reports from Code Analysis

Provides report generation infrastructure using Jinja2 templates:
- Executive Summary: High-level overview for stakeholders
- Technical Audit: Detailed analysis for developers
- Compliance Report: Regulatory/standards compliance

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import html
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from jinja2 import Environment, BaseLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None

from .code_metrics import ProjectMetrics, FileMetrics, ClassMetrics, ComplexityLevel
from .dependencies import DependencyGraph, DependencyStats


class ReportFormat(str, Enum):
    """Output format for reports."""
    HTML = "html"
    PDF = "pdf"


class ReportType(str, Enum):
    """Type of report to generate."""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_AUDIT = "technical_audit"
    COMPLIANCE = "compliance"


class ComplianceStandard(str, Enum):
    """Compliance standards for code analysis."""
    SONARQUBE = "sonarqube"
    OWASP = "owasp"
    ISO_25010 = "iso_25010"
    CUSTOM = "custom"


@dataclass
class Finding:
    """A single finding/issue in the report."""
    id: str
    title: str
    severity: str  # critical, high, medium, low, info
    category: str
    description: str
    location: Optional[str] = None
    recommendation: Optional[str] = None
    effort: Optional[str] = None  # hours to fix


@dataclass
class ReportSection:
    """A section within a report."""
    title: str
    content: str
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    charts: List[str] = field(default_factory=list)  # HTML chart snippets


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Code Analysis Report"
    project_name: str = "Project"
    author: str = "RAGIX Analysis"
    date: Optional[datetime] = None
    format: ReportFormat = ReportFormat.HTML
    include_charts: bool = True
    include_recommendations: bool = True
    include_appendix: bool = False
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None


@dataclass
class MavenData:
    """Maven project data for reports."""
    projects: List[Dict[str, Any]] = field(default_factory=list)
    total_dependencies: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    has_parent: bool = False
    is_multi_module: bool = False


@dataclass
class SonarData:
    """SonarQube data for reports."""
    project_key: str = ""
    quality_gate: str = "NONE"
    bugs: int = 0
    vulnerabilities: int = 0
    code_smells: int = 0
    coverage: Optional[float] = None
    duplication: Optional[float] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    hotspots: int = 0


@dataclass
class ReportData:
    """Data container for report generation."""
    config: ReportConfig
    metrics: Optional[ProjectMetrics] = None
    graph: Optional[DependencyGraph] = None
    maven: Optional[MavenData] = None
    sonar: Optional[SonarData] = None
    findings: List[Finding] = field(default_factory=list)
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base CSS Styles for Reports
# =============================================================================

REPORT_BASE_CSS = """
@page {
    size: A4;
    margin: 2cm;
    @top-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9px;
        color: #666;
    }
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
    line-height: 1.5;
    color: #333;
    background: #fff;
}

.report-container {
    max-width: 210mm;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.report-header {
    border-bottom: 3px solid #e94560;
    padding-bottom: 20px;
    margin-bottom: 30px;
}

.report-title {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 8px;
}

.report-subtitle {
    font-size: 16px;
    color: #666;
}

.report-meta {
    display: flex;
    gap: 30px;
    margin-top: 15px;
    font-size: 11px;
    color: #888;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Executive Summary */
.executive-summary {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 8px;
    padding: 25px;
    margin-bottom: 30px;
}

.summary-title {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 15px;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
}

.summary-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.summary-value {
    font-size: 32px;
    font-weight: 700;
    color: #e94560;
}

.summary-label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Health Score */
.health-score {
    display: flex;
    align-items: center;
    gap: 20px;
    margin: 20px 0;
}

.score-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.score-excellent { background: linear-gradient(135deg, #10b981, #059669); }
.score-good { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.score-fair { background: linear-gradient(135deg, #f59e0b, #d97706); }
.score-poor { background: linear-gradient(135deg, #ef4444, #dc2626); }

.score-details h3 {
    font-size: 16px;
    margin-bottom: 5px;
}

.score-details p {
    font-size: 12px;
    color: #666;
}

/* Sections */
.report-section {
    margin-bottom: 40px;
    page-break-inside: avoid;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #e94560;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

.section-content {
    font-size: 12px;
    line-height: 1.7;
}

/* Tables */
.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 11px;
}

.data-table th {
    background: #1a1a2e;
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
}

.data-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e9ecef;
}

.data-table tr:hover {
    background: #f8f9fa;
}

.data-table tr:nth-child(even) {
    background: #fafafa;
}

/* Findings */
.findings-list {
    margin: 20px 0;
}

.finding-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    margin-bottom: 15px;
    overflow: hidden;
}

.finding-header {
    display: flex;
    align-items: center;
    padding: 12px 15px;
    gap: 12px;
}

.severity-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.severity-critical { background: #fee2e2; color: #dc2626; }
.severity-high { background: #ffedd5; color: #ea580c; }
.severity-medium { background: #fef3c7; color: #d97706; }
.severity-low { background: #dbeafe; color: #2563eb; }
.severity-info { background: #e5e7eb; color: #6b7280; }

.finding-title {
    font-weight: 600;
    flex: 1;
}

.finding-body {
    padding: 0 15px 15px;
    font-size: 12px;
    color: #555;
}

.finding-location {
    font-family: "SF Mono", Monaco, monospace;
    font-size: 10px;
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 3px;
    color: #666;
}

.finding-recommendation {
    margin-top: 10px;
    padding: 10px;
    background: #f0fdf4;
    border-radius: 4px;
    border-left: 3px solid #10b981;
}

/* Charts placeholder */
.chart-container {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    min-height: 200px;
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin: 20px 0;
}

.metric-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 15px;
}

.metric-name {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
}

.metric-trend {
    font-size: 11px;
    margin-top: 5px;
}

.trend-up { color: #10b981; }
.trend-down { color: #ef4444; }
.trend-neutral { color: #6b7280; }

/* Progress Bars */
.progress-bar {
    height: 8px;
    background: #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.progress-good { background: #10b981; }
.progress-warning { background: #f59e0b; }
.progress-danger { background: #ef4444; }

/* Footer */
.report-footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e9ecef;
    font-size: 10px;
    color: #888;
    text-align: center;
}

.report-footer a {
    color: #e94560;
    text-decoration: none;
}

/* Print optimizations */
@media print {
    .report-container {
        padding: 0;
    }

    .finding-card, .report-section {
        page-break-inside: avoid;
    }

    .section-title {
        page-break-after: avoid;
    }
}
"""


# =============================================================================
# Template Strings
# =============================================================================

EXECUTIVE_SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>{{ base_css }}</style>
    {% if config.custom_css %}<style>{{ config.custom_css }}</style>{% endif %}
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1 class="report-title">{{ config.title }}</h1>
            <p class="report-subtitle">{{ config.project_name }} - Executive Summary</p>
            <div class="report-meta">
                <span class="meta-item">📅 {{ date }}</span>
                <span class="meta-item">👤 {{ config.author }}</span>
                <span class="meta-item">📊 Generated by RAGIX</span>
            </div>
        </header>

        <section class="executive-summary">
            <h2 class="summary-title">Key Metrics at a Glance</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value">{{ summary.total_files }}</div>
                    <div class="summary-label">Files Analyzed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{{ summary.total_loc | number_format }}</div>
                    <div class="summary-label">Lines of Code</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{{ summary.total_classes }}</div>
                    <div class="summary-label">Classes</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{{ summary.total_methods }}</div>
                    <div class="summary-label">Methods</div>
                </div>
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Overall Health Score</h2>
            <div class="health-score">
                <div class="score-circle score-{{ health_class }}">{{ health_score }}</div>
                <div class="score-details">
                    <h3>{{ health_label }}</h3>
                    <p>{{ health_description }}</p>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-name">Code Quality</div>
                    <div class="metric-value">{{ quality_score }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-{{ quality_class }}" style="width: {{ quality_score }}%"></div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Maintainability</div>
                    <div class="metric-value">{{ maintainability_score }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-{{ maintainability_class }}" style="width: {{ maintainability_score }}%"></div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Class Documentation</div>
                    <div class="metric-value">{{ class_doc_coverage }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-{{ class_doc_class }}" style="width: {{ class_doc_coverage }}%"></div>
                    </div>
                    <div style="font-size: 9px; color: #888; margin-top: 4px;">{{ summary.documented_classes }}/{{ summary.total_classes }} classes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Method Documentation</div>
                    <div class="metric-value">{{ method_doc_coverage }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-{{ method_doc_class }}" style="width: {{ method_doc_coverage }}%"></div>
                    </div>
                </div>
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Critical Findings</h2>
            <div class="findings-list">
                {% for finding in critical_findings %}
                <div class="finding-card">
                    <div class="finding-header">
                        <span class="severity-badge severity-{{ finding.severity }}">{{ finding.severity }}</span>
                        <span class="finding-title">{{ finding.title }}</span>
                        {% if finding.location %}<span class="finding-location">{{ finding.location }}</span>{% endif %}
                    </div>
                    <div class="finding-body">
                        <p>{{ finding.description }}</p>
                        {% if finding.recommendation %}
                        <div class="finding-recommendation">
                            <strong>Recommendation:</strong> {{ finding.recommendation }}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                {% if not critical_findings %}
                <p style="text-align: center; color: #888; padding: 20px;">No critical findings detected. ✓</p>
                {% endif %}
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Recommendations</h2>
            <div class="section-content">
                <ol>
                    {% for rec in recommendations %}
                    <li style="margin-bottom: 10px;">{{ rec }}</li>
                    {% endfor %}
                </ol>
            </div>
        </section>

        {% if maven %}
        <section class="report-section">
            <h2 class="section-title">Maven Dependencies</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-name">Projects</div>
                    <div class="metric-value">{{ maven.projects | length }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Dependencies</div>
                    <div class="metric-value">{{ maven.total_dependencies }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Conflicts</div>
                    <div class="metric-value" style="color: {{ '#ef4444' if maven.conflicts else 'inherit' }}">{{ maven.conflicts | length }}</div>
                </div>
            </div>
            {% if maven.conflicts %}
            <div class="finding-card" style="border-left: 3px solid #ef4444;">
                <div class="finding-header">
                    <span class="severity-badge severity-high">Warning</span>
                    <span class="finding-title">Dependency Version Conflicts</span>
                </div>
                <div class="finding-body">
                    <p>{{ maven.conflicts | length }} dependency conflict(s) detected. Multiple versions of the same artifact may cause runtime issues.</p>
                </div>
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if sonar %}
        <section class="report-section">
            <h2 class="section-title">SonarQube Quality Gate</h2>
            <div style="margin-bottom: 20px;">
                <span style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: bold;
                    background: {{ '#238636' if sonar.quality_gate == 'OK' else '#f85149' if sonar.quality_gate == 'ERROR' else '#d29922' }}; color: white;">
                    {{ sonar.quality_gate }}
                </span>
                <span style="margin-left: 10px; color: #888;">Project: {{ sonar.project_key }}</span>
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-name">Bugs</div>
                    <div class="metric-value" style="color: {{ '#ef4444' if sonar.bugs > 0 else 'inherit' }}">{{ sonar.bugs }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Vulnerabilities</div>
                    <div class="metric-value" style="color: {{ '#ef4444' if sonar.vulnerabilities > 0 else 'inherit' }}">{{ sonar.vulnerabilities }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Code Smells</div>
                    <div class="metric-value">{{ sonar.code_smells }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Coverage</div>
                    <div class="metric-value">{{ sonar.coverage or 'N/A' }}{{ '%' if sonar.coverage else '' }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Duplication</div>
                    <div class="metric-value">{{ sonar.duplication or 'N/A' }}{{ '%' if sonar.duplication else '' }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Security Hotspots</div>
                    <div class="metric-value">{{ sonar.hotspots }}</div>
                </div>
            </div>
        </section>
        {% endif %}

        <footer class="report-footer">
            <p>Generated by RAGIX Code Analysis Engine</p>
            <p>Author: <a href="mailto:olivier.vitrac@adservio.fr">Olivier Vitrac, PhD, HDR</a> | Adservio</p>
        </footer>
    </div>
</body>
</html>
"""

TECHNICAL_AUDIT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>{{ base_css }}</style>
    {% if config.custom_css %}<style>{{ config.custom_css }}</style>{% endif %}
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1 class="report-title">{{ config.title }}</h1>
            <p class="report-subtitle">{{ config.project_name }} - Technical Audit Report</p>
            <div class="report-meta">
                <span class="meta-item">📅 {{ date }}</span>
                <span class="meta-item">👤 {{ config.author }}</span>
                <span class="meta-item">🔍 Comprehensive Analysis</span>
            </div>
        </header>

        <section class="report-section">
            <h2 class="section-title">Project Overview</h2>
            <table class="data-table">
                <tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>
                <tr>
                    <td>Total Files</td>
                    <td>{{ summary.total_files }}</td>
                    <td>{{ summary.files_assessment }}</td>
                </tr>
                <tr>
                    <td>Lines of Code</td>
                    <td>{{ summary.total_loc | number_format }}</td>
                    <td>{{ summary.loc_assessment }}</td>
                </tr>
                <tr>
                    <td>Classes</td>
                    <td>{{ summary.total_classes }}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Methods/Functions</td>
                    <td>{{ summary.total_methods }}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Average Complexity</td>
                    <td>{{ summary.avg_complexity | round(2) }}</td>
                    <td>{{ summary.complexity_assessment }}</td>
                </tr>
                <tr>
                    <td>Technical Debt</td>
                    <td>{{ summary.tech_debt_hours | round(1) }} hours</td>
                    <td>{{ summary.debt_assessment }}</td>
                </tr>
            </table>
        </section>

        <section class="report-section">
            <h2 class="section-title">Complexity Analysis</h2>
            <div class="section-content">
                <h3 style="margin: 15px 0 10px;">Complexity Distribution</h3>
                <table class="data-table">
                    <tr>
                        <th>Level</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Status</th>
                    </tr>
                    {% for level in complexity_levels %}
                    <tr>
                        <td>{{ level.name }}</td>
                        <td>{{ level.count }}</td>
                        <td>{{ level.percentage }}%</td>
                        <td>
                            <span class="severity-badge severity-{{ level.badge }}">{{ level.status }}</span>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="section-content">
                <h3 style="margin: 20px 0 10px;">High Complexity Methods (Top 10)</h3>
                <table class="data-table">
                    <tr><th>Method</th><th>File</th><th>Line</th><th>Complexity</th></tr>
                    {% for method in high_complexity_methods %}
                    <tr>
                        <td><code>{{ method.name }}</code></td>
                        <td>{{ method.file }}</td>
                        <td>{{ method.line }}</td>
                        <td><span class="severity-badge severity-{{ method.badge }}">{{ method.complexity }}</span></td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Dependency Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-name">Total Dependencies</div>
                    <div class="metric-value">{{ dep_stats.total }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Circular Dependencies</div>
                    <div class="metric-value" style="color: {{ 'red' if dep_stats.circular > 0 else 'inherit' }}">{{ dep_stats.circular }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Avg. Coupling</div>
                    <div class="metric-value">{{ dep_stats.avg_coupling | round(2) }}</div>
                </div>
            </div>

            {% if dep_stats.circular > 0 %}
            <div class="finding-card" style="margin-top: 20px;">
                <div class="finding-header">
                    <span class="severity-badge severity-high">Warning</span>
                    <span class="finding-title">Circular Dependencies Detected</span>
                </div>
                <div class="finding-body">
                    <p>{{ dep_stats.circular }} circular dependency chain(s) were found. These can lead to maintenance issues and should be refactored.</p>
                </div>
            </div>
            {% endif %}
        </section>

        <section class="report-section">
            <h2 class="section-title">All Findings</h2>
            <div class="findings-list">
                {% for finding in findings %}
                <div class="finding-card">
                    <div class="finding-header">
                        <span class="severity-badge severity-{{ finding.severity }}">{{ finding.severity }}</span>
                        <span class="finding-title">{{ finding.title }}</span>
                        <span style="color: #888; font-size: 10px;">{{ finding.category }}</span>
                    </div>
                    <div class="finding-body">
                        <p>{{ finding.description }}</p>
                        {% if finding.location %}
                        <p><span class="finding-location">📍 {{ finding.location }}</span></p>
                        {% endif %}
                        {% if finding.recommendation %}
                        <div class="finding-recommendation">
                            <strong>Recommendation:</strong> {{ finding.recommendation }}
                            {% if finding.effort %}<br><em>Estimated effort: {{ finding.effort }}</em>{% endif %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Files Analysis</h2>
            <table class="data-table">
                <tr>
                    <th>File</th>
                    <th>LOC</th>
                    <th>Classes</th>
                    <th>Methods</th>
                    <th>Avg. CC</th>
                    <th>Status</th>
                </tr>
                {% for file in files_analysis %}
                <tr>
                    <td><code>{{ file.name }}</code></td>
                    <td>{{ file.loc }}</td>
                    <td>{{ file.classes }}</td>
                    <td>{{ file.methods }}</td>
                    <td>{{ file.avg_cc | round(1) }}</td>
                    <td><span class="severity-badge severity-{{ file.badge }}">{{ file.status }}</span></td>
                </tr>
                {% endfor %}
            </table>
        </section>

        {% if maven %}
        <section class="report-section">
            <h2 class="section-title">Maven Build Configuration</h2>
            {% for proj in maven.projects %}
            <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px;">{{ proj.gav }}</h3>
                <table class="data-table">
                    <tr><th>Scope</th><th>Dependencies</th></tr>
                    <tr><td>Compile</td><td>{{ proj.compile_deps }}</td></tr>
                    <tr><td>Test</td><td>{{ proj.test_deps }}</td></tr>
                </table>
            </div>
            {% endfor %}
            {% if maven.conflicts %}
            <h3 style="color: #ef4444; margin-top: 20px;">⚠ Dependency Conflicts</h3>
            <table class="data-table">
                <tr><th>Artifact</th><th>Conflicting Versions</th></tr>
                {% for c in maven.conflicts %}
                <tr>
                    <td>{{ c.artifact }}</td>
                    <td>{{ c.versions.keys() | list | join(', ') }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </section>
        {% endif %}

        {% if sonar %}
        <section class="report-section">
            <h2 class="section-title">SonarQube Analysis</h2>
            <div style="margin-bottom: 15px;">
                <span style="padding: 8px 16px; border-radius: 6px; font-weight: bold;
                    background: {{ '#238636' if sonar.quality_gate == 'OK' else '#f85149' if sonar.quality_gate == 'ERROR' else '#d29922' }}; color: white;">
                    Quality Gate: {{ sonar.quality_gate }}
                </span>
            </div>
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-name">Bugs</div><div class="metric-value">{{ sonar.bugs }}</div></div>
                <div class="metric-card"><div class="metric-name">Vulnerabilities</div><div class="metric-value">{{ sonar.vulnerabilities }}</div></div>
                <div class="metric-card"><div class="metric-name">Code Smells</div><div class="metric-value">{{ sonar.code_smells }}</div></div>
            </div>
            {% if sonar.issues %}
            <h3 style="margin-top: 20px;">Top Issues</h3>
            <table class="data-table">
                <tr><th>Severity</th><th>Rule</th><th>Location</th></tr>
                {% for issue in sonar.issues[:10] %}
                <tr>
                    <td><span class="severity-badge severity-{{ issue.severity | lower }}">{{ issue.severity }}</span></td>
                    <td>{{ issue.rule }}</td>
                    <td>{{ issue.file }}:{{ issue.line }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </section>
        {% endif %}

        <footer class="report-footer">
            <p>Generated by RAGIX Code Analysis Engine</p>
            <p>Author: <a href="mailto:olivier.vitrac@adservio.fr">Olivier Vitrac, PhD, HDR</a> | Adservio</p>
        </footer>
    </div>
</body>
</html>
"""

COMPLIANCE_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>{{ base_css }}</style>
    {% if config.custom_css %}<style>{{ config.custom_css }}</style>{% endif %}
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1 class="report-title">{{ config.title }}</h1>
            <p class="report-subtitle">{{ config.project_name }} - Compliance Report</p>
            <div class="report-meta">
                <span class="meta-item">📅 {{ date }}</span>
                <span class="meta-item">📋 Standard: {{ standard }}</span>
                <span class="meta-item">✅ Compliance: {{ compliance_score }}%</span>
            </div>
        </header>

        <section class="executive-summary">
            <h2 class="summary-title">Compliance Overview</h2>
            <div class="health-score">
                <div class="score-circle score-{{ compliance_class }}">{{ compliance_score }}%</div>
                <div class="score-details">
                    <h3>{{ compliance_label }}</h3>
                    <p>{{ compliance_description }}</p>
                </div>
            </div>
            <div class="summary-grid" style="margin-top: 20px;">
                <div class="summary-card">
                    <div class="summary-value" style="color: #10b981;">{{ rules_passed }}</div>
                    <div class="summary-label">Rules Passed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: #f59e0b;">{{ rules_warning }}</div>
                    <div class="summary-label">Warnings</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: #ef4444;">{{ rules_failed }}</div>
                    <div class="summary-label">Rules Failed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{{ rules_total }}</div>
                    <div class="summary-label">Total Rules</div>
                </div>
            </div>
        </section>

        <section class="report-section">
            <h2 class="section-title">Rule Compliance Details</h2>
            <table class="data-table">
                <tr>
                    <th>Rule ID</th>
                    <th>Rule Name</th>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Violations</th>
                </tr>
                {% for rule in rules %}
                <tr>
                    <td><code>{{ rule.id }}</code></td>
                    <td>{{ rule.name }}</td>
                    <td>{{ rule.category }}</td>
                    <td>
                        {% if rule.status == 'passed' %}
                        <span class="severity-badge severity-info" style="background:#d1fae5;color:#059669;">✓ Passed</span>
                        {% elif rule.status == 'warning' %}
                        <span class="severity-badge severity-medium">⚠ Warning</span>
                        {% else %}
                        <span class="severity-badge severity-critical">✗ Failed</span>
                        {% endif %}
                    </td>
                    <td>{{ rule.violations }}</td>
                </tr>
                {% endfor %}
            </table>
        </section>

        <section class="report-section">
            <h2 class="section-title">Violations by Category</h2>
            {% for category, violations in violations_by_category.items() %}
            <div style="margin-bottom: 25px;">
                <h3 style="font-size: 14px; margin-bottom: 10px; color: #1a1a2e;">
                    {{ category }} ({{ violations | length }} issues)
                </h3>
                <div class="findings-list">
                    {% for v in violations[:5] %}
                    <div class="finding-card">
                        <div class="finding-header">
                            <span class="severity-badge severity-{{ v.severity }}">{{ v.severity }}</span>
                            <span class="finding-title">{{ v.rule_id }}: {{ v.title }}</span>
                        </div>
                        <div class="finding-body">
                            <p>{{ v.description }}</p>
                            {% if v.location %}
                            <p><span class="finding-location">📍 {{ v.location }}</span></p>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                    {% if violations | length > 5 %}
                    <p style="text-align: center; color: #888; font-size: 11px;">
                        ... and {{ violations | length - 5 }} more violations in this category
                    </p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </section>

        <section class="report-section">
            <h2 class="section-title">Remediation Plan</h2>
            <div class="section-content">
                <p style="margin-bottom: 15px;">
                    Based on the analysis, the following remediation steps are recommended in priority order:
                </p>
                <ol>
                    {% for step in remediation_steps %}
                    <li style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
                        <strong>{{ step.title }}</strong>
                        <p style="color: #666; margin-top: 5px;">{{ step.description }}</p>
                        <p style="font-size: 11px; color: #888; margin-top: 5px;">
                            Effort: {{ step.effort }} | Impact: {{ step.impact }}
                        </p>
                    </li>
                    {% endfor %}
                </ol>
            </div>
        </section>

        <footer class="report-footer">
            <p>Generated by RAGIX Code Analysis Engine</p>
            <p>Author: <a href="mailto:olivier.vitrac@adservio.fr">Olivier Vitrac, PhD, HDR</a> | Adservio</p>
            <p style="margin-top: 10px; font-size: 9px;">
                This report is generated automatically and should be reviewed by qualified personnel.
            </p>
        </footer>
    </div>
</body>
</html>
"""


# =============================================================================
# Report Generators
# =============================================================================

class BaseReportGenerator(ABC):
    """Base class for report generators."""

    def __init__(self):
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for report generation. Install with: pip install jinja2")

        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml'])
        )
        # Add custom filters
        self.env.filters['number_format'] = lambda x: f"{x:,}" if isinstance(x, (int, float)) else x
        self.env.filters['round'] = lambda x, n=0: round(x, n) if isinstance(x, (int, float)) else x

    @abstractmethod
    def generate(self, data: ReportData) -> str:
        """Generate report HTML."""
        pass

    def to_pdf(self, html_content: str, output_path: Path) -> bool:
        """Convert HTML to PDF using WeasyPrint."""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("weasyprint is required for PDF generation. Install with: pip install weasyprint")

        try:
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(output_path)
            return True
        except Exception as e:
            print(f"PDF generation failed: {e}")
            return False

    def _get_health_class(self, score: int) -> Tuple[str, str, str]:
        """Get health score class, label, and description."""
        if score >= 80:
            return "excellent", "Excellent", "The codebase demonstrates strong quality practices."
        elif score >= 60:
            return "good", "Good", "The codebase is in good shape with minor improvements needed."
        elif score >= 40:
            return "fair", "Fair", "Several areas need attention to improve code quality."
        else:
            return "poor", "Needs Improvement", "Significant refactoring is recommended."

    def _get_progress_class(self, value: float) -> str:
        """Get progress bar class based on value."""
        if value >= 70:
            return "good"
        elif value >= 40:
            return "warning"
        return "danger"


class ExecutiveSummaryGenerator(BaseReportGenerator):
    """Generate executive summary reports."""

    def generate(self, data: ReportData) -> str:
        """Generate executive summary HTML."""
        template = self.env.from_string(EXECUTIVE_SUMMARY_TEMPLATE)

        # Calculate health score (0-100)
        health_score = self._calculate_health_score(data)
        health_class, health_label, health_desc = self._get_health_class(health_score)

        # Calculate sub-scores
        quality_score = self._calculate_quality_score(data)
        maintainability_score = self._calculate_maintainability_score(data)
        doc_coverage = data.summary.get("doc_coverage", 0)
        class_doc_coverage = data.summary.get("class_doc_coverage", 0)
        method_doc_coverage = data.summary.get("method_doc_coverage", 0)

        # Get critical findings
        critical_findings = [f for f in data.findings if f.severity in ("critical", "high")][:5]

        # Generate recommendations
        recommendations = self._generate_recommendations(data)

        return template.render(
            config=data.config,
            date=data.config.date.strftime("%Y-%m-%d") if data.config.date else datetime.now().strftime("%Y-%m-%d"),
            base_css=REPORT_BASE_CSS,
            summary=data.summary,
            health_score=health_score,
            health_class=health_class,
            health_label=health_label,
            health_description=health_desc,
            quality_score=quality_score,
            quality_class=self._get_progress_class(quality_score),
            maintainability_score=maintainability_score,
            maintainability_class=self._get_progress_class(maintainability_score),
            doc_coverage=doc_coverage,
            doc_class=self._get_progress_class(doc_coverage),
            class_doc_coverage=class_doc_coverage,
            class_doc_class=self._get_progress_class(class_doc_coverage),
            method_doc_coverage=method_doc_coverage,
            method_doc_class=self._get_progress_class(method_doc_coverage),
            critical_findings=critical_findings,
            recommendations=recommendations,
            maven=data.maven,
            sonar=data.sonar
        )

    def _calculate_health_score(self, data: ReportData) -> int:
        """Calculate overall health score."""
        scores = []

        # Complexity factor
        avg_cc = data.summary.get("avg_complexity", 5)
        complexity_score = max(0, 100 - (avg_cc - 1) * 10)
        scores.append(complexity_score)

        # Documentation factor
        doc_coverage = data.summary.get("doc_coverage", 0)
        scores.append(doc_coverage)

        # Issues factor
        critical = sum(1 for f in data.findings if f.severity == "critical")
        high = sum(1 for f in data.findings if f.severity == "high")
        issue_penalty = critical * 15 + high * 8
        issue_score = max(0, 100 - issue_penalty)
        scores.append(issue_score)

        return int(sum(scores) / len(scores)) if scores else 50

    def _calculate_quality_score(self, data: ReportData) -> int:
        """Calculate code quality score."""
        avg_cc = data.summary.get("avg_complexity", 5)
        return int(max(0, min(100, 100 - (avg_cc - 2) * 8)))

    def _calculate_maintainability_score(self, data: ReportData) -> int:
        """Calculate maintainability score."""
        avg_loc = data.summary.get("avg_file_loc", 200)
        # Penalize files over 300 LOC
        loc_penalty = max(0, (avg_loc - 300) / 10) if avg_loc > 300 else 0
        return int(max(0, min(100, 90 - loc_penalty)))

    def _generate_recommendations(self, data: ReportData) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        if data.summary.get("avg_complexity", 0) > 10:
            recommendations.append("Refactor methods with cyclomatic complexity > 10 to improve testability")

        if data.summary.get("doc_coverage", 100) < 50:
            recommendations.append("Improve documentation coverage to at least 70% for better maintainability")

        critical = sum(1 for f in data.findings if f.severity == "critical")
        if critical > 0:
            recommendations.append(f"Address {critical} critical issue(s) immediately to reduce technical risk")

        if data.summary.get("circular_deps", 0) > 0:
            recommendations.append("Eliminate circular dependencies to improve modularity")

        # Maven recommendations
        if data.maven and data.maven.conflicts:
            recommendations.append(f"Resolve {len(data.maven.conflicts)} Maven dependency version conflict(s)")

        # SonarQube recommendations
        if data.sonar:
            if data.sonar.bugs > 0:
                recommendations.append(f"Fix {data.sonar.bugs} bug(s) detected by SonarQube")
            if data.sonar.vulnerabilities > 0:
                recommendations.append(f"Address {data.sonar.vulnerabilities} security vulnerability/ies urgently")
            if data.sonar.coverage is not None and data.sonar.coverage < 60:
                recommendations.append(f"Increase test coverage from {data.sonar.coverage}% to at least 60%")

        if not recommendations:
            recommendations.append("Continue maintaining current code quality standards")
            recommendations.append("Consider increasing test coverage for critical paths")

        return recommendations


class TechnicalAuditGenerator(BaseReportGenerator):
    """Generate detailed technical audit reports."""

    def generate(self, data: ReportData) -> str:
        """Generate technical audit HTML."""
        template = self.env.from_string(TECHNICAL_AUDIT_TEMPLATE)

        # Prepare complexity distribution
        complexity_levels = self._get_complexity_distribution(data)

        # Get high complexity methods
        high_complexity_methods = self._get_high_complexity_methods(data)

        # Dependency statistics
        dep_stats = self._get_dependency_stats(data)

        # Files analysis
        files_analysis = self._get_files_analysis(data)

        return template.render(
            config=data.config,
            date=data.config.date.strftime("%Y-%m-%d") if data.config.date else datetime.now().strftime("%Y-%m-%d"),
            base_css=REPORT_BASE_CSS,
            summary=self._enrich_summary(data.summary),
            complexity_levels=complexity_levels,
            high_complexity_methods=high_complexity_methods,
            dep_stats=dep_stats,
            findings=data.findings,
            files_analysis=files_analysis,
            maven=data.maven,
            sonar=data.sonar
        )

    def _enrich_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Add assessments to summary."""
        result = dict(summary)

        # Files assessment
        total_files = summary.get("total_files", 0)
        if total_files < 50:
            result["files_assessment"] = "Small project"
        elif total_files < 200:
            result["files_assessment"] = "Medium project"
        else:
            result["files_assessment"] = "Large project"

        # LOC assessment
        total_loc = summary.get("total_loc", 0)
        if total_loc < 10000:
            result["loc_assessment"] = "Small codebase"
        elif total_loc < 50000:
            result["loc_assessment"] = "Medium codebase"
        else:
            result["loc_assessment"] = "Large codebase"

        # Complexity assessment
        avg_cc = summary.get("avg_complexity", 0)
        if avg_cc <= 5:
            result["complexity_assessment"] = "✓ Low complexity"
        elif avg_cc <= 10:
            result["complexity_assessment"] = "⚠ Moderate complexity"
        else:
            result["complexity_assessment"] = "✗ High complexity"

        # Debt assessment
        debt_hours = summary.get("tech_debt_hours", 0)
        if debt_hours < 40:
            result["debt_assessment"] = "✓ Low debt (< 1 week)"
        elif debt_hours < 160:
            result["debt_assessment"] = "⚠ Moderate debt (< 1 month)"
        else:
            result["debt_assessment"] = "✗ High debt (> 1 month)"

        return result

    def _get_complexity_distribution(self, data: ReportData) -> List[Dict[str, Any]]:
        """Get complexity distribution data."""
        # Default distribution if no metrics
        levels = [
            {"name": "Simple (1-5)", "count": 0, "percentage": 0, "badge": "info", "status": "OK"},
            {"name": "Moderate (6-10)", "count": 0, "percentage": 0, "badge": "low", "status": "OK"},
            {"name": "Complex (11-20)", "count": 0, "percentage": 0, "badge": "medium", "status": "Review"},
            {"name": "Very Complex (>20)", "count": 0, "percentage": 0, "badge": "high", "status": "Refactor"},
        ]

        if data.metrics:
            dist = data.summary.get("complexity_distribution", {})
            total = sum(dist.values()) or 1

            for level in levels:
                key = level["name"].split()[0].lower()
                count = dist.get(key, 0)
                level["count"] = count
                level["percentage"] = round(count / total * 100, 1)

        return levels

    def _get_high_complexity_methods(self, data: ReportData) -> List[Dict[str, Any]]:
        """Get list of high complexity methods."""
        methods = data.summary.get("high_complexity_methods", [])

        result = []
        for m in methods[:10]:
            cc = m.get("complexity", 0)
            badge = "info"
            if cc > 20:
                badge = "critical"
            elif cc > 15:
                badge = "high"
            elif cc > 10:
                badge = "medium"

            result.append({
                "name": m.get("name", "unknown"),
                "file": m.get("file", ""),
                "line": m.get("line", 0),
                "complexity": cc,
                "badge": badge
            })

        return result

    def _get_dependency_stats(self, data: ReportData) -> Dict[str, Any]:
        """Get dependency statistics."""
        return {
            "total": data.summary.get("total_deps", 0),
            "circular": data.summary.get("circular_deps", 0),
            "avg_coupling": data.summary.get("avg_coupling", 0)
        }

    def _get_files_analysis(self, data: ReportData) -> List[Dict[str, Any]]:
        """Get per-file analysis data."""
        files = data.summary.get("files", [])

        result = []
        for f in files[:20]:  # Top 20 files
            avg_cc = f.get("avg_cc", 0)
            if avg_cc <= 5:
                badge, status = "info", "Good"
            elif avg_cc <= 10:
                badge, status = "low", "OK"
            elif avg_cc <= 15:
                badge, status = "medium", "Review"
            else:
                badge, status = "high", "Refactor"

            result.append({
                "name": f.get("name", ""),
                "loc": f.get("loc", 0),
                "classes": f.get("classes", 0),
                "methods": f.get("methods", 0),
                "avg_cc": avg_cc,
                "badge": badge,
                "status": status
            })

        return result


class ComplianceReportGenerator(BaseReportGenerator):
    """Generate compliance reports."""

    def __init__(self, standard: ComplianceStandard = ComplianceStandard.SONARQUBE):
        super().__init__()
        self.standard = standard
        self.rules = self._load_rules(standard)

    def _load_rules(self, standard: ComplianceStandard) -> List[Dict[str, Any]]:
        """Load compliance rules for the standard."""
        # Default SonarQube-like rules
        return [
            {"id": "S001", "name": "Cyclomatic Complexity", "category": "Complexity", "threshold": 10},
            {"id": "S002", "name": "Method Length", "category": "Maintainability", "threshold": 100},
            {"id": "S003", "name": "Class Length", "category": "Maintainability", "threshold": 500},
            {"id": "S004", "name": "Documentation Coverage", "category": "Documentation", "threshold": 50},
            {"id": "S005", "name": "Circular Dependencies", "category": "Architecture", "threshold": 0},
            {"id": "S006", "name": "Code Duplication", "category": "Duplication", "threshold": 5},
            {"id": "S007", "name": "Coupling Between Objects", "category": "Architecture", "threshold": 20},
            {"id": "S008", "name": "Depth of Inheritance", "category": "Architecture", "threshold": 5},
            {"id": "S009", "name": "Comment Density", "category": "Documentation", "threshold": 10},
            {"id": "S010", "name": "Technical Debt Ratio", "category": "Debt", "threshold": 5},
        ]

    def generate(self, data: ReportData) -> str:
        """Generate compliance report HTML."""
        template = self.env.from_string(COMPLIANCE_REPORT_TEMPLATE)

        # Evaluate rules
        evaluated_rules = self._evaluate_rules(data)

        # Calculate compliance score
        passed = sum(1 for r in evaluated_rules if r["status"] == "passed")
        warning = sum(1 for r in evaluated_rules if r["status"] == "warning")
        failed = sum(1 for r in evaluated_rules if r["status"] == "failed")
        total = len(evaluated_rules)

        compliance_score = int((passed + warning * 0.5) / total * 100) if total > 0 else 100
        compliance_class, compliance_label, compliance_desc = self._get_health_class(compliance_score)

        # Group violations by category
        violations_by_category = self._group_violations(data.findings)

        # Generate remediation steps
        remediation_steps = self._generate_remediation(evaluated_rules, data)

        return template.render(
            config=data.config,
            date=data.config.date.strftime("%Y-%m-%d") if data.config.date else datetime.now().strftime("%Y-%m-%d"),
            base_css=REPORT_BASE_CSS,
            standard=self.standard.value.upper(),
            compliance_score=compliance_score,
            compliance_class=compliance_class,
            compliance_label=compliance_label,
            compliance_description=compliance_desc,
            rules_passed=passed,
            rules_warning=warning,
            rules_failed=failed,
            rules_total=total,
            rules=evaluated_rules,
            violations_by_category=violations_by_category,
            remediation_steps=remediation_steps
        )

    def _evaluate_rules(self, data: ReportData) -> List[Dict[str, Any]]:
        """Evaluate each rule against the data."""
        results = []

        for rule in self.rules:
            result = {
                "id": rule["id"],
                "name": rule["name"],
                "category": rule["category"],
                "violations": 0,
                "status": "passed"
            }

            # Check each rule
            if rule["id"] == "S001":  # Complexity
                high_cc = sum(1 for m in data.summary.get("high_complexity_methods", [])
                             if m.get("complexity", 0) > rule["threshold"])
                result["violations"] = high_cc
                if high_cc > 10:
                    result["status"] = "failed"
                elif high_cc > 0:
                    result["status"] = "warning"

            elif rule["id"] == "S004":  # Documentation
                doc_cov = data.summary.get("doc_coverage", 0)
                if doc_cov < rule["threshold"]:
                    result["status"] = "failed"
                    result["violations"] = 1
                elif doc_cov < rule["threshold"] + 20:
                    result["status"] = "warning"

            elif rule["id"] == "S005":  # Circular deps
                circular = data.summary.get("circular_deps", 0)
                result["violations"] = circular
                if circular > rule["threshold"]:
                    result["status"] = "failed"

            results.append(result)

        return results

    def _group_violations(self, findings: List[Finding]) -> Dict[str, List[Dict[str, Any]]]:
        """Group violations by category."""
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for f in findings:
            category = f.category
            if category not in groups:
                groups[category] = []

            groups[category].append({
                "rule_id": f.id,
                "title": f.title,
                "severity": f.severity,
                "description": f.description,
                "location": f.location
            })

        return groups

    def _generate_remediation(
        self,
        rules: List[Dict[str, Any]],
        data: ReportData
    ) -> List[Dict[str, Any]]:
        """Generate remediation plan."""
        steps = []

        # Sort by severity of violation
        failed_rules = [r for r in rules if r["status"] == "failed"]

        for rule in failed_rules:
            step = {
                "title": f"Fix {rule['name']} violations",
                "description": f"Address {rule['violations']} violations of rule {rule['id']}",
                "effort": "Medium",
                "impact": "High"
            }
            steps.append(step)

        if not steps:
            steps.append({
                "title": "Maintain current standards",
                "description": "All compliance rules are passing. Continue monitoring.",
                "effort": "Low",
                "impact": "Preventive"
            })

        return steps[:5]  # Top 5 remediation steps


# =============================================================================
# Main Report Engine
# =============================================================================

class ReportEngine:
    """Main report generation engine."""

    def __init__(self):
        self._generators: Dict[ReportType, BaseReportGenerator] = {}

    def register_generator(
        self,
        report_type: ReportType,
        generator: BaseReportGenerator
    ) -> None:
        """Register a report generator."""
        self._generators[report_type] = generator

    def generate(
        self,
        report_type: ReportType,
        data: ReportData,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a report."""
        # Get or create generator
        if report_type not in self._generators:
            if report_type == ReportType.EXECUTIVE_SUMMARY:
                self._generators[report_type] = ExecutiveSummaryGenerator()
            elif report_type == ReportType.TECHNICAL_AUDIT:
                self._generators[report_type] = TechnicalAuditGenerator()
            elif report_type == ReportType.COMPLIANCE:
                self._generators[report_type] = ComplianceReportGenerator()
            else:
                raise ValueError(f"Unknown report type: {report_type}")

        generator = self._generators[report_type]

        # Generate HTML
        html_content = generator.generate(data)

        # Output
        if output_path:
            if data.config.format == ReportFormat.PDF:
                generator.to_pdf(html_content, output_path)
            else:
                output_path.write_text(html_content)

        return html_content

    @staticmethod
    def create_report_data(
        config: ReportConfig,
        metrics: Optional[ProjectMetrics] = None,
        graph: Optional[DependencyGraph] = None
    ) -> ReportData:
        """Create ReportData from analysis results."""
        data = ReportData(config=config, metrics=metrics, graph=graph)

        # Build summary from metrics
        if metrics:
            # Use ProjectMetrics properties
            data.summary = {
                "total_files": metrics.total_files,
                "total_loc": metrics.total_code_lines,
                "total_classes": metrics.total_classes,
                "total_methods": metrics.total_methods,  # Includes class methods + standalone functions
                "avg_complexity": metrics.avg_complexity_per_method,
                "doc_coverage": int(metrics.doc_coverage),  # Balanced (50% class + 50% method)
                "class_doc_coverage": int(metrics.class_doc_coverage),  # Class-level Javadoc coverage
                "method_doc_coverage": int(metrics.method_doc_coverage),  # Method-level Javadoc coverage
                "documented_classes": metrics.documented_classes,
                "documented_methods": metrics.documented_methods,
                "tech_debt_hours": metrics.estimated_debt_hours,
                "maintainability_index": metrics.maintainability_index,
            }

            # Build high complexity methods list
            hotspots = metrics.get_hotspots(20)
            data.summary["high_complexity_methods"] = [
                {"name": name, "complexity": cc, "file": "", "line": 0}
                for name, cc in hotspots if cc > 10
            ]

            # Build files list for technical audit
            data.summary["files"] = [
                {
                    "name": Path(f.path).name,
                    "loc": f.code_lines,
                    "classes": f.class_count,
                    "methods": f.function_count + sum(c.method_count for c in f.class_metrics),
                    "avg_cc": f.total_complexity / max(1, f.function_count + sum(c.method_count for c in f.class_metrics)),
                }
                for f in metrics.file_metrics[:20]
            ]
        else:
            data.summary = {
                "total_files": 0,
                "total_loc": 0,
                "total_classes": 0,
                "total_methods": 0,
                "avg_complexity": 0,
                "doc_coverage": 0,
                "tech_debt_hours": 0,
                "maintainability_index": 100,
                "high_complexity_methods": [],
                "files": [],
            }

        # Build summary from graph
        if graph:
            stats = graph.get_stats()
            data.summary.update({
                "total_deps": stats.total_dependencies,
                "circular_deps": len(stats.cycles) if hasattr(stats, 'cycles') else 0,
                "avg_coupling": stats.avg_outgoing if hasattr(stats, 'avg_outgoing') else 0,
            })

        return data


def get_report_engine() -> ReportEngine:
    """Get singleton report engine instance."""
    return ReportEngine()


def generate_executive_summary(
    metrics: Optional[ProjectMetrics] = None,
    graph: Optional[DependencyGraph] = None,
    project_name: str = "Project",
    output_path: Optional[Path] = None,
    format: ReportFormat = ReportFormat.HTML
) -> str:
    """Convenience function to generate executive summary."""
    config = ReportConfig(
        title="Executive Summary",
        project_name=project_name,
        format=format,
        date=datetime.now()
    )

    data = ReportEngine.create_report_data(config, metrics, graph)
    engine = get_report_engine()

    return engine.generate(ReportType.EXECUTIVE_SUMMARY, data, output_path)


def generate_technical_audit(
    metrics: Optional[ProjectMetrics] = None,
    graph: Optional[DependencyGraph] = None,
    project_name: str = "Project",
    output_path: Optional[Path] = None,
    format: ReportFormat = ReportFormat.HTML
) -> str:
    """Convenience function to generate technical audit."""
    config = ReportConfig(
        title="Technical Audit Report",
        project_name=project_name,
        format=format,
        date=datetime.now()
    )

    data = ReportEngine.create_report_data(config, metrics, graph)
    engine = get_report_engine()

    return engine.generate(ReportType.TECHNICAL_AUDIT, data, output_path)


def generate_compliance_report(
    metrics: Optional[ProjectMetrics] = None,
    graph: Optional[DependencyGraph] = None,
    project_name: str = "Project",
    standard: ComplianceStandard = ComplianceStandard.SONARQUBE,
    output_path: Optional[Path] = None,
    format: ReportFormat = ReportFormat.HTML
) -> str:
    """Convenience function to generate compliance report."""
    config = ReportConfig(
        title="Compliance Report",
        project_name=project_name,
        format=format,
        date=datetime.now()
    )

    data = ReportEngine.create_report_data(config, metrics, graph)
    engine = get_report_engine()
    engine.register_generator(
        ReportType.COMPLIANCE,
        ComplianceReportGenerator(standard)
    )

    return engine.generate(ReportType.COMPLIANCE, data, output_path)
