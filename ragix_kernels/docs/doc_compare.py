"""
Kernel: Document Compare
Stage: 3 (Synthesis)

Performs inter-document comparison to detect discrepancies (écarts).
Identifies:
- Missing cross-references
- Terminology inconsistencies
- Version mismatches
- Overlapping/conflicting requirements

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import logging
import json
import re
from difflib import SequenceMatcher

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# =============================================================================
# Semantic Filtering for Terminology Variation Detection
# =============================================================================

# French verb conjugation endings (to identify verb forms)
FRENCH_VERB_ENDINGS = {
    "er", "ir", "re",  # infinitive
    "e", "es", "ent", "ons", "ez",  # present
    "é", "ée", "és", "ées",  # past participle
    "ait", "aient", "ions", "iez",  # imperfect
    "era", "eront", "erons", "erez",  # future
    "ant",  # present participle
}

# French noun/adjective plural and gender endings
FRENCH_INFLECTION_PATTERNS = {
    # (singular_suffix, plural_suffix) pairs
    ("", "s"),
    ("", "x"),
    ("al", "aux"),
    ("ail", "aux"),
    ("eu", "eux"),
    ("eau", "eaux"),
    # Gender variations
    ("", "e"),
    ("eur", "euse"),
    ("teur", "trice"),
    ("f", "ve"),
    ("el", "elle"),
    ("en", "enne"),
    ("on", "onne"),
}

# Words that should NOT be grouped despite edit distance similarity
# Key: word, Value: set of words it should NOT be grouped with
SEMANTIC_EXCLUSIONS = {
    # French: different meanings despite similar spelling
    "outils": {"utiles", "utile"},  # tools ≠ useful
    "utiles": {"outils", "outil"},  # useful ≠ tools
    "rapports": {"apportés", "apporte", "apporter"},  # reports ≠ brought
    "apportés": {"rapports", "rapport"},  # brought ≠ reports
    "rapport": {"apporter", "apporté"},
    "partie": {"partir", "part"},  # part/side ≠ to leave
    "partir": {"partie", "parties"},  # to leave ≠ parts
    "mode": {"modem"},  # mode ≠ modem
    "modem": {"mode", "modes"},
    "site": {"suite", "siste"},  # site ≠ suite
    "suite": {"site", "sites"},
    "helle": {"échelle", "elle"},  # nonsense word
    "elle": {"échelle", "helle"},  # she ≠ scale
    "échelle": {"elle", "helle"},  # scale ≠ she/nonsense
    "auteur": {"acteur"},  # author ≠ actor
    "acteur": {"auteur", "auteurs"},
    "état": {"étant"},  # state ≠ being (gerund)
    "notion": {"option", "options"},  # notion ≠ option
    "option": {"notion", "notions"},
    "cause": {"cases"},  # cause ≠ cases (English)
    "cases": {"cause", "causes"},  # English word
    "interne": {"internet"},  # internal ≠ internet
    "internet": {"interne", "internes"},
    "entire": {"entrée", "entrer", "entière"},  # English ≠ French
    "entrée": {"entire"},
    "proper": {"propre"},  # English ≠ French
    "propre": {"proper", "properly"},
    # Mixed language exclusions
    "export": {"report"},  # export ≠ report
    "report": {"export", "exporté", "exporter"},  # English report ≠ French export
    "contents": {"contient"},  # English ≠ French
    "contient": {"contents"},
    "test": {"tests", "tester"},  # Often English in French docs
    "command": {"commande"},  # English ≠ French
    "commande": {"command", "commands"},
    "status": {"statut", "statuts"},  # English ≠ French
    "statut": {"status"},
    "require": {"requise"},  # English ≠ French
    "requise": {"require", "required"},
    "clear": {"cleared", "clearer"},  # English only
    "update": {"updates", "updated"},  # English only
    "release": {"releases", "released"},  # English only
    "filter": {"filters", "filtres"},  # Mixed
    "filtres": {"filter", "filters"},
    "list": {"liste", "listés"},  # Mixed
    "liste": {"list"},
    "pass": {"passe", "passed"},  # Mixed
    "passe": {"pass", "passed"},
    "object": {"objet"},  # Mixed
    "objet": {"object"},
    "import": {"importe", "importer"},  # Can be both languages
}

# Minimum word length to consider (filter out fragments)
MIN_TERM_LENGTH = 4

# Maximum edit distance ratio for semantic grouping
SEMANTIC_SIMILARITY_THRESHOLD = 0.85


def get_french_lemma(word: str) -> str:
    """
    Get approximate lemma for French word (rule-based, no external deps).
    Returns the likely root form for grouping purposes.
    """
    word = word.lower().strip()

    # Too short to lemmatize
    if len(word) < 4:
        return word

    # Try verb infinitive reconstruction
    for ending in ["ées", "és", "ée", "é"]:  # past participle → infinitive
        if word.endswith(ending):
            stem = word[:-len(ending)]
            if len(stem) >= 3:
                return stem + "er"

    # Try plural → singular
    if word.endswith("aux") and len(word) > 4:
        return word[:-3] + "al"
    if word.endswith("eux") and len(word) > 4:
        return word[:-3] + "eu"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("x"):
        return word[:-1]

    # Try feminine → masculine
    if word.endswith("elle") and len(word) > 5:
        return word[:-4] + "el"
    if word.endswith("enne") and len(word) > 5:
        return word[:-4] + "en"
    if word.endswith("euse") and len(word) > 5:
        return word[:-4] + "eur"
    if word.endswith("trice") and len(word) > 6:
        return word[:-5] + "teur"

    return word


def is_valid_french_term(word: str) -> bool:
    """
    Check if word looks like a valid French/technical term.
    Filters out fragments and obvious noise.
    """
    if len(word) < MIN_TERM_LENGTH:
        return False

    # Must have at least one vowel
    vowels = set("aeiouyàâäéèêëïîôùûü")
    if not any(c in vowels for c in word.lower()):
        return False

    # Filter common fragments
    fragments = {"tion", "ment", "ille", "elle", "onne", "ette", "ière", "stion"}
    if word.lower() in fragments:
        return False

    # Filter if more than 50% consonants in a row
    consonant_run = 0
    max_consonant_run = 0
    for c in word.lower():
        if c not in vowels and c.isalpha():
            consonant_run += 1
            max_consonant_run = max(max_consonant_run, consonant_run)
        else:
            consonant_run = 0
    if max_consonant_run > len(word) * 0.6:
        return False

    return True


def are_semantically_related(t1: str, t2: str) -> bool:
    """
    Check if two terms are semantically related (not just similar spelling).
    Returns False for known false positives.
    """
    t1_lower = t1.lower()
    t2_lower = t2.lower()

    # Check explicit exclusions
    if t1_lower in SEMANTIC_EXCLUSIONS:
        if t2_lower in SEMANTIC_EXCLUSIONS[t1_lower]:
            return False
    if t2_lower in SEMANTIC_EXCLUSIONS:
        if t1_lower in SEMANTIC_EXCLUSIONS[t2_lower]:
            return False

    # Check if same lemma (likely grammatical variants)
    lemma1 = get_french_lemma(t1_lower)
    lemma2 = get_french_lemma(t2_lower)

    if lemma1 == lemma2:
        return True

    # Check if one is prefix of other with grammatical suffix
    if t1_lower.startswith(t2_lower) or t2_lower.startswith(t1_lower):
        shorter = t1_lower if len(t1_lower) < len(t2_lower) else t2_lower
        longer = t2_lower if len(t1_lower) < len(t2_lower) else t1_lower
        suffix = longer[len(shorter):]

        # Valid French suffixes
        valid_suffixes = {"s", "e", "es", "é", "ée", "és", "ées", "er", "eur",
                         "euse", "ment", "tion", "ant", "ent"}
        if suffix in valid_suffixes:
            return True

    return False


class DocCompareKernel(Kernel):
    """
    Compare documents to detect discrepancies and inconsistencies.

    This kernel analyzes the document corpus to identify:
    - Missing references (document A mentions B, but B doesn't exist)
    - Broken references (cross-reference to non-existent section)
    - Terminology conflicts (same concept, different terms)
    - Version mismatches (references to outdated versions)
    - Requirement overlaps (similar requirements in different docs)
    - Coverage gaps (expected topics not covered)

    Configuration options:
        project.path: Path to the indexed project (required)
        similarity_threshold: Min similarity for overlap detection (default: 0.7)
        check_versions: Enable version mismatch detection (default: true)
        check_terminology: Enable terminology analysis (default: true)
        expected_coverage: List of expected concepts (default: [])

    Dependencies:
        doc_metadata: File inventory
        doc_extract: Key sentences
        doc_func_extract: Functionality catalog (optional)
        doc_concepts: Concept mappings

    Output:
        discrepancies: List of detected issues
        reference_graph: Document reference network
        terminology_clusters: Groups of related terms
        coverage_analysis: Expected vs actual coverage
        statistics: Analysis statistics
    """

    name = "doc_compare"
    version = "1.0.0"
    category = "docs"
    stage = 3
    description = "Detect inter-document discrepancies and inconsistencies"

    requires = ["doc_metadata", "doc_extract", "doc_concepts"]
    provides = ["discrepancies", "reference_graph", "terminology_analysis"]

    # Patterns for reference detection
    DOC_REF_PATTERNS = [
        re.compile(r"SPD-(\d+)", re.IGNORECASE),
        re.compile(r"ParisSURF4-([A-Z]+-\d+)", re.IGNORECASE),
        re.compile(r"(?:cf\.|voir|see|ref\.|référence)\s+([^\s,]+)", re.IGNORECASE),
        re.compile(r"document\s+[«\"']([^»\"']+)[»\"']", re.IGNORECASE),
    ]

    VERSION_PATTERN = re.compile(r"v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Perform inter-document comparison analysis."""
        # Get configuration
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        similarity_threshold = input.config.get("similarity_threshold", 0.7)
        check_versions = input.config.get("check_versions", True)
        check_terminology = input.config.get("check_terminology", True)
        expected_coverage = input.config.get("expected_coverage", [])

        logger.info("[doc_compare] Starting inter-document comparison")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        extract_path = input.dependencies.get("doc_extract")
        concepts_path = input.dependencies.get("doc_concepts")
        func_path = input.dependencies.get("doc_func_extract")

        if not all(p and p.exists() for p in [metadata_path, extract_path, concepts_path]):
            raise RuntimeError("Missing required dependencies")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(extract_path) as f:
            extract_data = json.load(f).get("data", {})
        with open(concepts_path) as f:
            concepts_data = json.load(f).get("data", {})

        # Optional functionality data
        func_data = {}
        if func_path and func_path.exists():
            with open(func_path) as f:
                func_data = json.load(f).get("data", {})

        # Build document inventory
        files = {f["file_id"]: f for f in metadata_data.get("files", [])}
        file_names = {f["file_id"]: Path(f["path"]).name for f in metadata_data.get("files", [])}
        file_paths = {f["file_id"]: f["path"] for f in metadata_data.get("files", [])}

        # Build searchable document index
        doc_index = self._build_doc_index(files)

        # Collect discrepancies
        discrepancies: List[Dict[str, Any]] = []

        # 1. Check cross-references
        logger.info("[doc_compare] Checking cross-references...")
        ref_graph, ref_issues = self._check_references(
            extract_data.get("by_file", {}),
            doc_index,
            file_names
        )
        discrepancies.extend(ref_issues)

        # 2. Check functionality cross-references
        if func_data:
            func_issues = self._check_func_references(func_data, doc_index)
            discrepancies.extend(func_issues)

        # 3. Check terminology consistency
        terminology_clusters = {}
        if check_terminology:
            logger.info("[doc_compare] Analyzing terminology...")
            terminology_clusters, term_issues = self._check_terminology(
                extract_data.get("by_file", {}),
                concepts_data.get("concepts", [])
            )
            discrepancies.extend(term_issues)

        # 4. Check version consistency
        if check_versions:
            logger.info("[doc_compare] Checking version references...")
            version_issues = self._check_versions(
                extract_data.get("by_file", {}),
                file_names
            )
            discrepancies.extend(version_issues)

        # 5. Detect overlapping requirements
        logger.info("[doc_compare] Detecting overlaps...")
        overlap_issues = self._detect_overlaps(
            extract_data.get("by_file", {}),
            similarity_threshold
        )
        discrepancies.extend(overlap_issues)

        # 6. Coverage analysis
        coverage = self._analyze_coverage(
            concepts_data.get("concepts", []),
            expected_coverage
        )

        # Categorize discrepancies
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        for d in discrepancies:
            by_type[d.get("type", "unknown")].append(d)
            by_severity[d.get("severity", "info")].append(d)

        # Statistics
        statistics = {
            "total_discrepancies": len(discrepancies),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "documents_analyzed": len(files),
            "references_found": sum(len(v) for v in ref_graph.values()),
            "terminology_clusters": len(terminology_clusters),
        }

        logger.info(
            f"[doc_compare] Found {len(discrepancies)} discrepancies "
            f"({statistics['by_severity'].get('error', 0)} errors, "
            f"{statistics['by_severity'].get('warning', 0)} warnings)"
        )

        return {
            "discrepancies": discrepancies,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "reference_graph": ref_graph,
            "terminology_clusters": terminology_clusters,
            "coverage_analysis": coverage,
            "statistics": statistics,
        }

    def _build_doc_index(self, files: Dict[str, Dict]) -> Dict[str, str]:
        """Build searchable index of document names to file_ids."""
        index = {}
        for file_id, file_info in files.items():
            path = file_info.get("path", "")
            name = Path(path).name
            stem = Path(path).stem

            # Index by various forms
            index[name.lower()] = file_id
            index[stem.lower()] = file_id

            # Extract SPD number if present
            spd_match = re.search(r"SPD-(\d+)", name, re.IGNORECASE)
            if spd_match:
                index[f"spd-{spd_match.group(1)}"] = file_id

        return index

    def _check_references(
        self,
        extracts: Dict[str, Dict],
        doc_index: Dict[str, str],
        file_names: Dict[str, str]
    ) -> Tuple[Dict[str, List[str]], List[Dict]]:
        """Check document cross-references."""
        ref_graph: Dict[str, List[str]] = defaultdict(list)
        issues: List[Dict] = []

        for file_id, file_data in extracts.items():
            sentences = file_data.get("sentences", [])
            source_name = file_names.get(file_id, file_id)

            for sent in sentences:
                text = sent.get("text", "")

                for pattern in self.DOC_REF_PATTERNS:
                    for match in pattern.finditer(text):
                        ref_text = match.group(1) if match.lastindex else match.group(0)
                        ref_key = ref_text.lower()

                        # Check if reference exists
                        if ref_key in doc_index:
                            ref_graph[file_id].append(doc_index[ref_key])
                        elif ref_key.startswith("spd-"):
                            # SPD reference to non-existent document
                            issues.append({
                                "type": "missing_reference",
                                "severity": "warning",
                                "source_file_id": file_id,
                                "source_name": source_name,
                                "target": ref_text.upper(),
                                "context": text[:100],
                                "description": f"Reference to non-existent document: {ref_text.upper()}"
                            })

        return dict(ref_graph), issues

    def _check_func_references(
        self,
        func_data: Dict[str, Any],
        doc_index: Dict[str, str]
    ) -> List[Dict]:
        """Check functionality cross-references."""
        issues = []

        missing_refs = func_data.get("missing_references", [])
        for ref in missing_refs:
            issues.append({
                "type": "missing_spd_reference",
                "severity": "error",
                "source": ref.get("source", ""),
                "target": ref.get("target", ""),
                "description": f"{ref.get('source')} references non-existent {ref.get('target')}"
            })

        return issues

    def _check_terminology(
        self,
        extracts: Dict[str, Dict],
        concepts: List[Dict]
    ) -> Tuple[Dict[str, List[str]], List[Dict]]:
        """
        Analyze terminology consistency with semantic filtering.

        Uses French morphology and explicit exclusion lists to avoid
        false positives from:
        - Different words with similar spelling (outils ≠ utiles)
        - Cross-language matches (export ≠ report)
        - Word fragments and noise
        """
        issues = []
        terminology_clusters = defaultdict(list)

        # Build term frequency per document
        doc_terms: Dict[str, Counter] = {}
        for file_id, file_data in extracts.items():
            terms = Counter()
            for sent in file_data.get("sentences", []):
                text = sent.get("text", "").lower()
                # Extract technical terms (simplified)
                words = re.findall(r'\b[a-zéèêëàâäùûü]{4,}\b', text)
                # Filter out invalid terms
                valid_words = [w for w in words if is_valid_french_term(w)]
                terms.update(valid_words)
            doc_terms[file_id] = terms

        # Find terms that appear with similar frequency but different spellings
        all_terms = Counter()
        for terms in doc_terms.values():
            all_terms.update(terms.keys())

        # Group similar terms with semantic validation
        term_list = list(all_terms.keys())
        used = set()

        for i, t1 in enumerate(term_list):
            if t1 in used:
                continue
            cluster = [t1]
            for t2 in term_list[i+1:]:
                if t2 in used:
                    continue
                # Check spelling similarity AND semantic relatedness
                if self._term_similar(t1, t2) and are_semantically_related(t1, t2):
                    cluster.append(t2)
                    used.add(t2)
            if len(cluster) > 1:
                terminology_clusters[t1] = cluster

        # Report significant terminology variations (true positives only)
        for base_term, variants in terminology_clusters.items():
            # Only report if we have real semantic variants (same concept, different forms)
            # Skip pure grammatical variations (singular/plural, conjugations)
            if len(variants) > 2:
                # Check if this is just grammatical variation
                lemmas = {get_french_lemma(v) for v in variants}
                if len(lemmas) > 1:  # Multiple lemmas = potential real inconsistency
                    issues.append({
                        "type": "terminology_variation",
                        "severity": "info",
                        "base_term": base_term,
                        "variants": variants,
                        "lemmas": list(lemmas),
                        "description": f"Multiple term variants: {', '.join(variants[:5])}"
                    })

        return dict(terminology_clusters), issues

    def _term_similar(self, t1: str, t2: str) -> bool:
        """
        Check if two terms are similar (potential variants).

        Uses edit distance but does NOT check semantic relatedness
        (that's done separately in _check_terminology).
        """
        if t1 == t2:
            return False

        # Filter out very short terms
        if len(t1) < MIN_TERM_LENGTH or len(t2) < MIN_TERM_LENGTH:
            return False

        # Check prefix match (e.g., "carrefour" vs "carrefours")
        if t1.startswith(t2) or t2.startswith(t1):
            return abs(len(t1) - len(t2)) <= 3

        # Check edit distance with stricter threshold
        if len(t1) <= 10 and len(t2) <= 10:
            ratio = SequenceMatcher(None, t1, t2).ratio()
            return ratio > SEMANTIC_SIMILARITY_THRESHOLD

        return False

    def _check_versions(
        self,
        extracts: Dict[str, Dict],
        file_names: Dict[str, str]
    ) -> List[Dict]:
        """Check for version inconsistencies."""
        issues = []

        # Extract version numbers from filenames
        file_versions: Dict[str, str] = {}
        for file_id, name in file_names.items():
            match = self.VERSION_PATTERN.search(name)
            if match:
                file_versions[file_id] = match.group(1)

        # Look for version references in content
        version_refs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for file_id, file_data in extracts.items():
            for sent in file_data.get("sentences", []):
                text = sent.get("text", "")
                for match in self.VERSION_PATTERN.finditer(text):
                    version = match.group(1)
                    version_refs[file_id].append((version, text[:50]))

        # Compare referenced versions with actual versions
        for file_id, refs in version_refs.items():
            actual = file_versions.get(file_id)
            for ref_ver, context in refs:
                if actual and ref_ver != actual:
                    # Potential version mismatch
                    issues.append({
                        "type": "version_reference",
                        "severity": "info",
                        "file_id": file_id,
                        "file_name": file_names.get(file_id, ""),
                        "actual_version": actual,
                        "referenced_version": ref_ver,
                        "context": context,
                        "description": f"References version {ref_ver}, file is {actual}"
                    })

        return issues

    def _detect_overlaps(
        self,
        extracts: Dict[str, Dict],
        threshold: float
    ) -> List[Dict]:
        """Detect overlapping content between documents."""
        issues = []

        # Compare sentences between different documents
        file_sentences: Dict[str, List[str]] = {}
        for file_id, file_data in extracts.items():
            sents = [s.get("text", "")[:200] for s in file_data.get("sentences", [])]
            file_sentences[file_id] = sents

        file_ids = list(file_sentences.keys())
        compared = set()

        for i, fid1 in enumerate(file_ids[:50]):  # Limit comparisons
            for fid2 in file_ids[i+1:50]:
                if (fid1, fid2) in compared:
                    continue
                compared.add((fid1, fid2))

                # Check sentence similarity
                overlaps = []
                for s1 in file_sentences[fid1][:10]:
                    for s2 in file_sentences[fid2][:10]:
                        if len(s1) > 50 and len(s2) > 50:
                            ratio = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
                            if ratio >= threshold:
                                overlaps.append((s1, s2, ratio))

                if overlaps:
                    issues.append({
                        "type": "content_overlap",
                        "severity": "info",
                        "file_1": fid1,
                        "file_2": fid2,
                        "overlap_count": len(overlaps),
                        "max_similarity": max(o[2] for o in overlaps),
                        "sample": overlaps[0][0][:100] if overlaps else "",
                        "description": f"{len(overlaps)} similar sentences between documents"
                    })

        return issues

    def _analyze_coverage(
        self,
        concepts: List[Dict],
        expected: List[str]
    ) -> Dict[str, Any]:
        """Analyze coverage of expected concepts."""
        actual_concepts = {c["label"].lower() for c in concepts}

        covered = []
        missing = []

        for exp in expected:
            exp_lower = exp.lower()
            if exp_lower in actual_concepts or any(exp_lower in ac for ac in actual_concepts):
                covered.append(exp)
            else:
                missing.append(exp)

        return {
            "expected": expected,
            "covered": covered,
            "missing": missing,
            "coverage_rate": len(covered) / len(expected) if expected else 1.0
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total = stats.get("total_discrepancies", 0)
        by_severity = stats.get("by_severity", {})
        errors = by_severity.get("error", 0)
        warnings = by_severity.get("warning", 0)

        return (
            f"Comparison: {total} discrepancies found "
            f"({errors} errors, {warnings} warnings). "
            f"Analyzed {stats.get('documents_analyzed', 0)} documents."
        )
