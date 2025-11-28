"""
SonarQube/SonarCloud Integration - Fetch code quality metrics

Connects to SonarQube or SonarCloud API to retrieve:
- Project metrics (bugs, vulnerabilities, code smells)
- Quality gate status
- Issues and hotspots
- Measure history
- Component details

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, quote

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class SonarSeverity(str, Enum):
    """Issue severity levels."""
    BLOCKER = "BLOCKER"
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class SonarIssueType(str, Enum):
    """Issue types."""
    BUG = "BUG"
    VULNERABILITY = "VULNERABILITY"
    CODE_SMELL = "CODE_SMELL"
    SECURITY_HOTSPOT = "SECURITY_HOTSPOT"


class QualityGateStatus(str, Enum):
    """Quality gate status."""
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"
    NONE = "NONE"


@dataclass
class SonarIssue:
    """A code quality issue from Sonar."""
    key: str
    rule: str
    severity: SonarSeverity
    issue_type: SonarIssueType
    message: str
    component: str
    line: Optional[int] = None
    effort: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: str = "OPEN"

    @property
    def file_path(self) -> str:
        """Extract file path from component."""
        # Component format: project:path/to/file.java
        if ":" in self.component:
            return self.component.split(":", 1)[1]
        return self.component


@dataclass
class SonarMetric:
    """A project metric from Sonar."""
    key: str
    value: str
    name: Optional[str] = None
    description: Optional[str] = None

    @property
    def numeric_value(self) -> Optional[float]:
        """Get numeric value if applicable."""
        try:
            return float(self.value)
        except (ValueError, TypeError):
            return None


@dataclass
class SonarProject:
    """A Sonar project summary."""
    key: str
    name: str
    qualifier: str = "TRK"
    visibility: str = "public"
    last_analysis: Optional[str] = None
    quality_gate: QualityGateStatus = QualityGateStatus.NONE
    metrics: Dict[str, SonarMetric] = field(default_factory=dict)
    issues: List[SonarIssue] = field(default_factory=list)

    @property
    def bugs(self) -> int:
        """Get bug count."""
        metric = self.metrics.get("bugs")
        return int(metric.numeric_value or 0) if metric else 0

    @property
    def vulnerabilities(self) -> int:
        """Get vulnerability count."""
        metric = self.metrics.get("vulnerabilities")
        return int(metric.numeric_value or 0) if metric else 0

    @property
    def code_smells(self) -> int:
        """Get code smell count."""
        metric = self.metrics.get("code_smells")
        return int(metric.numeric_value or 0) if metric else 0

    @property
    def coverage(self) -> Optional[float]:
        """Get test coverage percentage."""
        metric = self.metrics.get("coverage")
        return metric.numeric_value if metric else None

    @property
    def duplicated_lines_density(self) -> Optional[float]:
        """Get code duplication percentage."""
        metric = self.metrics.get("duplicated_lines_density")
        return metric.numeric_value if metric else None


class SonarClient:
    """
    Client for SonarQube/SonarCloud API.

    Usage:
        client = SonarClient(base_url="https://sonarcloud.io", token="...")
        project = client.get_project("my-project")
        issues = client.get_issues("my-project", severities=[SonarSeverity.CRITICAL])
    """

    # Common metrics to fetch
    DEFAULT_METRICS = [
        "bugs",
        "vulnerabilities",
        "code_smells",
        "security_hotspots",
        "coverage",
        "duplicated_lines_density",
        "ncloc",
        "complexity",
        "cognitive_complexity",
        "sqale_index",
        "sqale_debt_ratio",
        "reliability_rating",
        "security_rating",
        "sqale_rating",
    ]

    def __init__(
        self,
        base_url: str = "https://sonarcloud.io",
        token: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize Sonar client.

        Args:
            base_url: SonarQube/SonarCloud URL
            token: API token for authentication
            organization: Organization key (required for SonarCloud)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for Sonar integration")

        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("SONAR_TOKEN")
        self.organization = organization or os.environ.get("SONAR_ORGANIZATION")
        self._session = requests.Session()

        if self.token:
            self._session.auth = (self.token, "")

    def _api_get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an API GET request."""
        url = urljoin(self.base_url + "/", f"api/{endpoint}")
        params = params or {}

        if self.organization:
            params["organization"] = self.organization

        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_project(
        self,
        project_key: str,
        metrics: Optional[List[str]] = None,
    ) -> SonarProject:
        """
        Get project with metrics.

        Args:
            project_key: The project key in Sonar
            metrics: List of metrics to fetch (uses DEFAULT_METRICS if None)

        Returns:
            SonarProject with metrics populated
        """
        # Get component info
        try:
            component_data = self._api_get(
                "components/show",
                {"component": project_key}
            )
        except Exception as e:
            # Create minimal project if API fails
            return SonarProject(key=project_key, name=project_key)

        component = component_data.get("component", {})

        project = SonarProject(
            key=component.get("key", project_key),
            name=component.get("name", project_key),
            qualifier=component.get("qualifier", "TRK"),
            visibility=component.get("visibility", "public"),
        )

        # Get quality gate status
        try:
            qg_data = self._api_get(
                "qualitygates/project_status",
                {"projectKey": project_key}
            )
            status_str = qg_data.get("projectStatus", {}).get("status", "NONE")
            project.quality_gate = QualityGateStatus(status_str)
        except Exception:
            pass

        # Get metrics
        metrics = metrics or self.DEFAULT_METRICS
        try:
            measures_data = self._api_get(
                "measures/component",
                {
                    "component": project_key,
                    "metricKeys": ",".join(metrics),
                }
            )
            for measure in measures_data.get("component", {}).get("measures", []):
                metric = SonarMetric(
                    key=measure.get("metric"),
                    value=measure.get("value", "0"),
                )
                project.metrics[metric.key] = metric
        except Exception:
            pass

        return project

    def get_issues(
        self,
        project_key: str,
        severities: Optional[List[SonarSeverity]] = None,
        types: Optional[List[SonarIssueType]] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SonarIssue]:
        """
        Get issues for a project.

        Args:
            project_key: The project key
            severities: Filter by severity levels
            types: Filter by issue types
            statuses: Filter by statuses (OPEN, CONFIRMED, REOPENED, RESOLVED, CLOSED)
            limit: Maximum number of issues to return

        Returns:
            List of SonarIssue objects
        """
        params = {
            "componentKeys": project_key,
            "ps": min(limit, 500),  # API max is 500
        }

        if severities:
            params["severities"] = ",".join(s.value for s in severities)

        if types:
            params["types"] = ",".join(t.value for t in types)

        if statuses:
            params["statuses"] = ",".join(statuses)
        else:
            params["statuses"] = "OPEN,CONFIRMED,REOPENED"

        try:
            data = self._api_get("issues/search", params)
        except Exception:
            return []

        issues = []
        for item in data.get("issues", []):
            try:
                issue = SonarIssue(
                    key=item.get("key", ""),
                    rule=item.get("rule", ""),
                    severity=SonarSeverity(item.get("severity", "INFO")),
                    issue_type=SonarIssueType(item.get("type", "CODE_SMELL")),
                    message=item.get("message", ""),
                    component=item.get("component", ""),
                    line=item.get("line"),
                    effort=item.get("effort"),
                    tags=item.get("tags", []),
                    status=item.get("status", "OPEN"),
                )
                issues.append(issue)
            except (ValueError, KeyError):
                continue

        return issues

    def get_hotspots(
        self,
        project_key: str,
        status: str = "TO_REVIEW",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get security hotspots for a project.

        Args:
            project_key: The project key
            status: Hotspot status (TO_REVIEW, REVIEWED)
            limit: Maximum number to return

        Returns:
            List of hotspot dictionaries
        """
        params = {
            "projectKey": project_key,
            "status": status,
            "ps": min(limit, 500),
        }

        try:
            data = self._api_get("hotspots/search", params)
            return data.get("hotspots", [])
        except Exception:
            return []

    def get_measures_history(
        self,
        project_key: str,
        metrics: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical measures for a project.

        Args:
            project_key: The project key
            metrics: Metrics to fetch history for
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping metric names to list of {date, value} dicts
        """
        params = {
            "component": project_key,
            "metrics": ",".join(metrics),
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            data = self._api_get("measures/search_history", params)
        except Exception:
            return {}

        result = {}
        for measure in data.get("measures", []):
            metric = measure.get("metric")
            history = []
            for item in measure.get("history", []):
                history.append({
                    "date": item.get("date"),
                    "value": item.get("value"),
                })
            result[metric] = history

        return result

    def list_projects(
        self,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List projects in the organization.

        Args:
            query: Search query
            limit: Maximum number to return

        Returns:
            List of project dictionaries
        """
        params = {
            "ps": min(limit, 500),
        }

        if query:
            params["q"] = query

        try:
            data = self._api_get("projects/search", params)
            return data.get("components", [])
        except Exception:
            return []


@dataclass
class SonarReport:
    """Generate a Sonar quality report."""
    project: SonarProject
    issues: List[SonarIssue]
    hotspots: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the report."""
        # Count issues by severity
        by_severity = {}
        for issue in self.issues:
            sev = issue.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Count issues by type
        by_type = {}
        for issue in self.issues:
            typ = issue.issue_type.value
            by_type[typ] = by_type.get(typ, 0) + 1

        return {
            "project": self.project.key,
            "quality_gate": self.project.quality_gate.value,
            "metrics": {
                "bugs": self.project.bugs,
                "vulnerabilities": self.project.vulnerabilities,
                "code_smells": self.project.code_smells,
                "coverage": self.project.coverage,
                "duplication": self.project.duplicated_lines_density,
            },
            "issues": {
                "total": len(self.issues),
                "by_severity": by_severity,
                "by_type": by_type,
            },
            "hotspots": len(self.hotspots),
        }

    def to_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = []
        lines.append(f"# Sonar Quality Report: {self.project.name}")
        lines.append("")

        # Quality gate
        gate_emoji = {
            QualityGateStatus.OK: "✅",
            QualityGateStatus.WARN: "⚠️",
            QualityGateStatus.ERROR: "❌",
            QualityGateStatus.NONE: "➖",
        }
        emoji = gate_emoji.get(self.project.quality_gate, "➖")
        lines.append(f"**Quality Gate:** {emoji} {self.project.quality_gate.value}")
        lines.append("")

        # Metrics summary
        lines.append("## Metrics Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Bugs | {self.project.bugs} |")
        lines.append(f"| Vulnerabilities | {self.project.vulnerabilities} |")
        lines.append(f"| Code Smells | {self.project.code_smells} |")

        if self.project.coverage is not None:
            lines.append(f"| Coverage | {self.project.coverage:.1f}% |")

        if self.project.duplicated_lines_density is not None:
            lines.append(f"| Duplication | {self.project.duplicated_lines_density:.1f}% |")

        lines.append("")

        # Issues by severity
        if self.issues:
            lines.append("## Issues by Severity")
            lines.append("")

            severity_order = [
                SonarSeverity.BLOCKER,
                SonarSeverity.CRITICAL,
                SonarSeverity.MAJOR,
                SonarSeverity.MINOR,
                SonarSeverity.INFO,
            ]

            for sev in severity_order:
                issues = [i for i in self.issues if i.severity == sev]
                if issues:
                    lines.append(f"### {sev.value} ({len(issues)})")
                    lines.append("")
                    for issue in issues[:10]:
                        loc = f":{issue.line}" if issue.line else ""
                        lines.append(f"- **{issue.rule}** - {issue.message}")
                        lines.append(f"  - File: `{issue.file_path}{loc}`")
                    if len(issues) > 10:
                        lines.append(f"- ... and {len(issues) - 10} more")
                    lines.append("")

        return "\n".join(lines)


# Convenience functions

def connect_sonar(
    base_url: str = "https://sonarcloud.io",
    token: Optional[str] = None,
    organization: Optional[str] = None,
) -> SonarClient:
    """Create a Sonar client."""
    return SonarClient(base_url, token, organization)


def get_project_report(
    client: SonarClient,
    project_key: str,
) -> SonarReport:
    """Get a full quality report for a project."""
    project = client.get_project(project_key)
    issues = client.get_issues(project_key)
    hotspots = client.get_hotspots(project_key)

    return SonarReport(
        project=project,
        issues=issues,
        hotspots=hotspots,
    )
