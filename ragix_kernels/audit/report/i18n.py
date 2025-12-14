"""
Internationalization (i18n) for KOAS Reports.

Provides translations for:
- Section titles and headings
- Analysis descriptions and interpretations
- Recommendations
- Quality grades and risk levels
- Common terms

Technical table data remains in English.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class Language(Enum):
    """Supported languages."""
    EN = "en"
    FR = "fr"


# =============================================================================
# Translation Dictionaries
# =============================================================================

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Section titles
    "section.executive": {
        "en": "Executive Summary",
        "fr": "Synthèse Exécutive",
    },
    "section.overview": {
        "en": "Codebase Overview",
        "fr": "Vue d'Ensemble du Code Source",
    },
    "section.architecture": {
        "en": "Architecture Analysis",
        "fr": "Analyse Architecturale",
    },
    "section.quality": {
        "en": "Code Quality Assessment",
        "fr": "Évaluation de la Qualité du Code",
    },
    "section.risk": {
        "en": "Risk Assessment",
        "fr": "Évaluation des Risques",
    },
    "section.coupling": {
        "en": "Coupling Analysis",
        "fr": "Analyse du Couplage",
    },
    "section.debt": {
        "en": "Technical Debt Analysis",
        "fr": "Analyse de la Dette Technique",
    },
    "section.recommendations": {
        "en": "Recommendations",
        "fr": "Recommandations",
    },
    "section.methodology": {
        "en": "Methodology",
        "fr": "Méthodologie",
    },
    "section.appendix": {
        "en": "Appendix",
        "fr": "Annexes",
    },
    "section.toc": {
        "en": "Table of Contents",
        "fr": "Table des Matières",
    },
    "section.hotspots": {
        "en": "Complexity Hotspots",
        "fr": "Points Chauds de Complexité",
    },
    "section.drift": {
        "en": "Drift Analysis",
        "fr": "Analyse de la Dérive",
    },

    # Titles
    "title.audit_report": {
        "en": "Technical Audit Report",
        "fr": "Rapport d'Audit Technique",
    },

    # Subsections
    "subsection.key_findings": {
        "en": "Key Findings",
        "fr": "Résultats Clés",
    },
    "subsection.metrics_summary": {
        "en": "Metrics Summary",
        "fr": "Résumé des Métriques",
    },
    "subsection.risk_distribution": {
        "en": "Risk Distribution",
        "fr": "Distribution des Risques",
    },
    "subsection.critical_components": {
        "en": "Critical Components",
        "fr": "Composants Critiques",
    },
    "subsection.hotspots": {
        "en": "Complexity Hotspots",
        "fr": "Points Chauds de Complexité",
    },
    "subsection.dead_code": {
        "en": "Dead Code Analysis",
        "fr": "Analyse du Code Mort",
    },
    "subsection.coupling_zones": {
        "en": "Coupling Zone Distribution",
        "fr": "Distribution des Zones de Couplage",
    },
    "subsection.action_plan": {
        "en": "Action Plan",
        "fr": "Plan d'Action",
    },
    "subsection.analysis_metadata": {
        "en": "Analysis Metadata",
        "fr": "Métadonnées de l'Analyse",
    },
    "subsection.metrics_reference": {
        "en": "Metrics Reference",
        "fr": "Référence des Métriques",
    },

    # Quality grades
    "grade.excellent": {
        "en": "Excellent",
        "fr": "Excellent",
    },
    "grade.good": {
        "en": "Good",
        "fr": "Bon",
    },
    "grade.moderate": {
        "en": "Moderate",
        "fr": "Modéré",
    },
    "grade.poor": {
        "en": "Poor",
        "fr": "Insuffisant",
    },
    "grade.critical": {
        "en": "Critical",
        "fr": "Critique",
    },

    # Risk levels
    "risk.critical": {
        "en": "Critical",
        "fr": "Critique",
    },
    "risk.high": {
        "en": "High",
        "fr": "Élevé",
    },
    "risk.medium": {
        "en": "Medium",
        "fr": "Moyen",
    },
    "risk.low": {
        "en": "Low",
        "fr": "Faible",
    },

    # Priority levels
    "priority.immediate": {
        "en": "Immediate",
        "fr": "Immédiat",
    },
    "priority.short_term": {
        "en": "Short-term",
        "fr": "Court terme",
    },
    "priority.medium_term": {
        "en": "Medium-term",
        "fr": "Moyen terme",
    },
    "priority.long_term": {
        "en": "Long-term",
        "fr": "Long terme",
    },

    # Coupling zones (Martin metrics)
    "zone.pain": {
        "en": "Zone of Pain",
        "fr": "Zone de Douleur",
    },
    "zone.useless": {
        "en": "Zone of Uselessness",
        "fr": "Zone d'Inutilité",
    },
    "zone.main_sequence": {
        "en": "Main Sequence",
        "fr": "Séquence Principale",
    },
    "zone.balanced": {
        "en": "Balanced Zone",
        "fr": "Zone Équilibrée",
    },

    # Common labels
    "label.files": {
        "en": "Files",
        "fr": "Fichiers",
    },
    "label.classes": {
        "en": "Classes",
        "fr": "Classes",
    },
    "label.methods": {
        "en": "Methods",
        "fr": "Méthodes",
    },
    "label.functions": {
        "en": "Functions",
        "fr": "Fonctions",
    },
    "label.lines_of_code": {
        "en": "Lines of Code",
        "fr": "Lignes de Code",
    },
    "label.complexity": {
        "en": "Complexity",
        "fr": "Complexité",
    },
    "label.maintainability": {
        "en": "Maintainability Index",
        "fr": "Indice de Maintenabilité",
    },
    "label.technical_debt": {
        "en": "Technical Debt",
        "fr": "Dette Technique",
    },
    "label.person_days": {
        "en": "person-days",
        "fr": "jours-homme",
    },
    "label.hours": {
        "en": "hours",
        "fr": "heures",
    },
    "label.coverage": {
        "en": "Coverage",
        "fr": "Couverture",
    },
    "label.quality": {
        "en": "Quality",
        "fr": "Qualité",
    },
    "label.overall_grade": {
        "en": "Overall Grade",
        "fr": "Note Globale",
    },
    "label.critical_risk": {
        "en": "Critical Risk Components",
        "fr": "Composants à Risque Critique",
    },
    "label.high_risk": {
        "en": "High Risk Components",
        "fr": "Composants à Risque Élevé",
    },

    # Footer
    "footer.generated_by": {
        "en": "Generated by",
        "fr": "Généré par",
    },
    "footer.date": {
        "en": "Date",
        "fr": "Date",
    },
    "footer.confidential": {
        "en": "CONFIDENTIAL - Internal Use Only",
        "fr": "CONFIDENTIEL - Usage Interne Uniquement",
    },

    # Table captions
    "caption.overview_metrics": {
        "en": "Table: Overview Metrics",
        "fr": "Tableau : Métriques de Vue d'Ensemble",
    },
    "caption.quality_grades": {
        "en": "Table: Quality Grades",
        "fr": "Tableau : Notes de Qualité",
    },
    "caption.risk_distribution": {
        "en": "Table: Risk Distribution by Component",
        "fr": "Tableau : Distribution des Risques par Composant",
    },
    "caption.hotspots": {
        "en": "Table: Complexity Hotspots (Top 20)",
        "fr": "Tableau : Points Chauds de Complexité (Top 20)",
    },
    "caption.coupling_packages": {
        "en": "Table: Package Coupling Metrics",
        "fr": "Tableau : Métriques de Couplage par Package",
    },
    "caption.recommendations": {
        "en": "Table: Prioritized Recommendations",
        "fr": "Tableau : Recommandations Priorisées",
    },

    # Analysis interpretations
    "interp.high_complexity": {
        "en": "High complexity indicates methods that are difficult to understand and maintain. Consider refactoring into smaller, focused units.",
        "fr": "Une complexité élevée indique des méthodes difficiles à comprendre et à maintenir. Envisagez un refactoring en unités plus petites et focalisées.",
    },
    "interp.low_maintainability": {
        "en": "Low maintainability index suggests the codebase will be expensive to evolve. Prioritize refactoring and documentation improvements.",
        "fr": "Un indice de maintenabilité bas suggère que le code sera coûteux à faire évoluer. Priorisez le refactoring et l'amélioration de la documentation.",
    },
    "interp.high_coupling": {
        "en": "High coupling between components increases change propagation risk. Consider introducing abstractions or interfaces.",
        "fr": "Un couplage élevé entre composants augmente le risque de propagation des changements. Envisagez d'introduire des abstractions ou interfaces.",
    },
    "interp.dead_code": {
        "en": "Dead code candidates were detected. Review these for potential removal to reduce maintenance burden.",
        "fr": "Des candidats au code mort ont été détectés. Examinez-les pour une suppression potentielle afin de réduire la charge de maintenance.",
    },
    "interp.zone_pain": {
        "en": "Packages in the Zone of Pain are highly concrete and stable, making them difficult to change. Consider adding abstractions.",
        "fr": "Les packages dans la Zone de Douleur sont très concrets et stables, les rendant difficiles à modifier. Envisagez d'ajouter des abstractions.",
    },
    "interp.zone_useless": {
        "en": "Packages in the Zone of Uselessness are highly abstract but unused. Review for potential consolidation or removal.",
        "fr": "Les packages dans la Zone d'Inutilité sont très abstraits mais inutilisés. Examinez-les pour une consolidation ou suppression potentielle.",
    },
    "interp.technical_debt": {
        "en": "Technical debt estimation is based on complexity violations and code smells. Prioritize debt repayment in high-risk areas.",
        "fr": "L'estimation de la dette technique est basée sur les violations de complexité et les code smells. Priorisez le remboursement de la dette dans les zones à haut risque.",
    },
    "interp.mco_risk": {
        "en": "High MCO (Maintenance in Operational Conditions) risk indicates components that are costly to maintain and evolve.",
        "fr": "Un risque MCO (Maintien en Conditions Opérationnelles) élevé indique des composants coûteux à maintenir et à faire évoluer.",
    },

    # Recommendations templates
    "reco.refactor_complexity": {
        "en": "Refactor high-complexity methods in {component} (CC={cc}). Extract smaller helper methods to improve readability.",
        "fr": "Refactorisez les méthodes à haute complexité dans {component} (CC={cc}). Extrayez des méthodes auxiliaires plus petites pour améliorer la lisibilité.",
    },
    "reco.improve_documentation": {
        "en": "Improve documentation coverage for {component}. Current coverage: {coverage}%.",
        "fr": "Améliorez la couverture documentaire pour {component}. Couverture actuelle : {coverage}%.",
    },
    "reco.reduce_coupling": {
        "en": "Reduce coupling for package {package} (Ce={ce}, Ca={ca}). Consider dependency injection or interfaces.",
        "fr": "Réduisez le couplage pour le package {package} (Ce={ce}, Ca={ca}). Envisagez l'injection de dépendances ou des interfaces.",
    },
    "reco.remove_dead_code": {
        "en": "Review and remove dead code candidates in {component}. Estimated {count} unreachable elements.",
        "fr": "Examinez et supprimez les candidats au code mort dans {component}. Estimation de {count} éléments inaccessibles.",
    },
    "reco.address_debt": {
        "en": "Address technical debt in {component}. Estimated effort: {hours}h ({days} person-days).",
        "fr": "Traitez la dette technique dans {component}. Effort estimé : {hours}h ({days} jours-homme).",
    },
    "reco.stabilize_package": {
        "en": "Stabilize package {package} (I={instability}). High instability increases change propagation risk.",
        "fr": "Stabilisez le package {package} (I={instability}). Une instabilité élevée augmente le risque de propagation des changements.",
    },

    # Report metadata
    "meta.generated_by": {
        "en": "Generated by KOAS (Kernel-Orchestrated Audit System)",
        "fr": "Généré par KOAS (Système d'Audit Orchestré par Noyaux)",
    },
    "meta.confidential": {
        "en": "CONFIDENTIAL - Internal Use Only",
        "fr": "CONFIDENTIEL - Usage Interne Uniquement",
    },
    "meta.draft": {
        "en": "DRAFT",
        "fr": "BROUILLON",
    },
    "meta.final": {
        "en": "FINAL",
        "fr": "FINAL",
    },

    # Audit context
    "context.audit_scope": {
        "en": "Audit Scope",
        "fr": "Périmètre de l'Audit",
    },
    "context.audit_objectives": {
        "en": "Audit Objectives",
        "fr": "Objectifs de l'Audit",
    },
    "context.audit_questions": {
        "en": "Audit Questions",
        "fr": "Questions de l'Audit",
    },
    "context.key_findings": {
        "en": "Key Findings",
        "fr": "Conclusions Principales",
    },
    "context.recommendations_summary": {
        "en": "Recommendations Summary",
        "fr": "Résumé des Recommandations",
    },
}


# =============================================================================
# Translator Class
# =============================================================================

@dataclass
class I18N:
    """
    Internationalization handler for KOAS reports.

    Usage:
        i18n = I18N(language="fr")
        title = i18n.t("section.executive")  # "Synthese Executive"
        desc = i18n.t("reco.refactor_complexity", component="MyClass", cc=25)
    """

    language: str = "en"
    fallback: str = "en"

    def t(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key with optional interpolation.

        Args:
            key: Translation key (e.g., "section.executive")
            default: Default value if key not found
            **kwargs: Interpolation variables

        Returns:
            Translated string
        """
        translation = TRANSLATIONS.get(key, {})

        # Try requested language, then fallback, then default, then key
        text = (
            translation.get(self.language)
            or translation.get(self.fallback)
            or default
            or key
        )

        # Interpolate variables
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass  # Leave uninterpolated if variable missing

        return text

    def grade(self, letter: str) -> str:
        """Translate quality grade."""
        grade_map = {
            "A": "grade.excellent",
            "B": "grade.good",
            "C": "grade.moderate",
            "D": "grade.poor",
            "F": "grade.critical",
        }
        return self.t(grade_map.get(letter.upper(), "grade.moderate"))

    def risk_level(self, level: str) -> str:
        """Translate risk level."""
        level_map = {
            "critical": "risk.critical",
            "high": "risk.high",
            "medium": "risk.medium",
            "low": "risk.low",
        }
        return self.t(level_map.get(level.lower(), "risk.medium"))

    def priority(self, priority: str) -> str:
        """Translate priority level."""
        priority_map = {
            "immediate": "priority.immediate",
            "short_term": "priority.short_term",
            "short-term": "priority.short_term",
            "medium_term": "priority.medium_term",
            "medium-term": "priority.medium_term",
            "long_term": "priority.long_term",
            "long-term": "priority.long_term",
        }
        return self.t(priority_map.get(priority.lower(), "priority.medium_term"))

    def zone(self, zone_name: str) -> str:
        """Translate coupling zone name."""
        zone_map = {
            "pain": "zone.pain",
            "useless": "zone.useless",
            "main_sequence": "zone.main_sequence",
            "balanced": "zone.balanced",
        }
        return self.t(zone_map.get(zone_name.lower(), zone_name))


def get_translator(language: str = "en") -> I18N:
    """Get translator instance for specified language."""
    return I18N(language=language)
