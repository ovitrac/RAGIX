"""
Kernel: Document Functionality Extraction
Stage: 2 (Analysis)

Extracts structured functionality definitions from specification documents.
Specifically targets SPD (Spécifications de Processus Détaillées) documents
to build a functionality catalog with clear descriptions.

Uses LLM to extract:
- Functionality IDs and names
- Clear descriptions
- Actors and triggers
- Dependencies and references

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import logging
import json
import time
import re

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocFuncExtractKernel(Kernel):
    """
    Extract structured functionalities from specification documents.

    This kernel identifies and extracts functionality definitions from
    SPD (functional specification) documents, producing a structured
    catalog that can be used for gap analysis and cross-referencing.

    Configuration options:
        project.path: Path to the indexed project (required)
        llm_model: Ollama model name (default: "granite3.1-moe:3b")
        language: Output language "fr" or "en" (default: "fr")
        spd_pattern: Regex to identify SPD files (default: r"SPD-\d+")
        include_non_spd: Also extract from non-SPD docs (default: false)

    Dependencies:
        doc_metadata: File inventory
        doc_extract: Key sentences/extracts
        doc_structure: Document sections

    Output:
        functionalities: List of extracted functionalities
        by_document: Functionalities grouped by source document
        by_category: Functionalities grouped by functional category
        cross_references: SPD cross-reference graph
        statistics: Extraction statistics
    """

    name = "doc_func_extract"
    version = "1.0.0"
    category = "docs"
    stage = 2
    description = "Extract structured functionalities from specification documents"

    requires = ["doc_metadata", "doc_extract", "doc_structure"]
    provides = ["functionalities", "functionality_catalog", "cross_references"]

    # SPD identification patterns
    SPD_PATTERN = re.compile(r"SPD-(\d+)", re.IGNORECASE)
    FUNC_ID_PATTERN = re.compile(r"(SPD-\d+-F\d+|FUNC-\d+|F\d+\.\d+)", re.IGNORECASE)

    # Extraction prompt templates
    PROMPT_FR = """Tu es un expert en ingénierie des exigences. Analyse ce document de spécification et extrais les fonctionnalités définies.

**Document:** {title}
**Type:** SPD (Spécification de Processus Détaillée)

**Sections du document:**
{sections}

**Contenu extrait:**
{extracts}

**Instructions:**
Identifie chaque fonctionnalité distincte et fournis:
1. ID: identifiant unique (format: SPD-XX-FYY)
2. NOM: nom court de la fonctionnalité
3. DESCRIPTION: description claire en 1-2 phrases
4. ACTEURS: qui utilise cette fonctionnalité
5. DÉCLENCHEUR: qu'est-ce qui initie cette fonctionnalité
6. RÉFÉRENCES: autres SPD ou documents mentionnés

**Format de réponse (une fonctionnalité par bloc, séparés par ---):**
ID: SPD-XX-F01
NOM: [nom]
DESCRIPTION: [description]
ACTEURS: [acteur1, acteur2]
DÉCLENCHEUR: [événement ou action]
RÉFÉRENCES: [SPD-YY, SPD-ZZ]
---"""

    PROMPT_EN = """You are a requirements engineering expert. Analyze this specification document and extract the defined functionalities.

**Document:** {title}
**Type:** SPD (Detailed Process Specification)

**Document sections:**
{sections}

**Extracted content:**
{extracts}

**Instructions:**
Identify each distinct functionality and provide:
1. ID: unique identifier (format: SPD-XX-FYY)
2. NAME: short functionality name
3. DESCRIPTION: clear description in 1-2 sentences
4. ACTORS: who uses this functionality
5. TRIGGER: what initiates this functionality
6. REFERENCES: other SPD or documents mentioned

**Response format (one functionality per block, separated by ---):**
ID: SPD-XX-F01
NAME: [name]
DESCRIPTION: [description]
ACTORS: [actor1, actor2]
TRIGGER: [event or action]
REFERENCES: [SPD-YY, SPD-ZZ]
---"""

    # Functional categories for classification
    CATEGORIES = {
        "interface": ["interface", "communication", "protocole", "api", "échange"],
        "monitoring": ["surveillance", "monitoring", "alarme", "alerte", "état"],
        "control": ["commande", "contrôle", "pilotage", "gestion", "command"],
        "data": ["données", "export", "import", "historique", "data", "opendata"],
        "configuration": ["configuration", "paramètre", "édition", "edition"],
        "display": ["affichage", "visualisation", "carte", "plan", "display"],
        "security": ["sécurité", "authentification", "accès", "security"],
        "maintenance": ["maintenance", "diagnostic", "journal", "log"],
    }

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Extract functionalities from specification documents."""
        import ollama

        # Get configuration
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        llm_model = input.config.get("llm_model", "granite3.1-moe:3b")
        language = input.config.get("language", "fr")
        spd_pattern = input.config.get("spd_pattern", r"SPD-\d+")
        include_non_spd = input.config.get("include_non_spd", False)

        prompt_template = self.PROMPT_FR if language == "fr" else self.PROMPT_EN
        spd_regex = re.compile(spd_pattern, re.IGNORECASE)

        logger.info(f"[doc_func_extract] Extracting functionalities with {llm_model}")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        extract_path = input.dependencies.get("doc_extract")
        structure_path = input.dependencies.get("doc_structure")

        if not all(p and p.exists() for p in [metadata_path, extract_path]):
            raise RuntimeError("Missing required dependencies")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(extract_path) as f:
            extract_data = json.load(f).get("data", {})
        with open(structure_path) as f:
            structure_data = json.load(f).get("data", {})

        # Identify SPD documents
        files = {f["file_id"]: f for f in metadata_data.get("files", [])}
        spd_files = {}
        non_spd_files = {}

        for file_id, file_info in files.items():
            file_path = file_info.get("path", "")
            if spd_regex.search(file_path):
                spd_files[file_id] = file_info
            elif include_non_spd:
                non_spd_files[file_id] = file_info

        logger.info(f"[doc_func_extract] Found {len(spd_files)} SPD documents")

        # Process SPD documents
        all_functionalities: List[Dict[str, Any]] = []
        by_document: Dict[str, List[Dict]] = defaultdict(list)
        cross_references: Dict[str, List[str]] = defaultdict(list)
        errors: List[Dict[str, str]] = []

        extracts = extract_data.get("by_file", {})
        structures = structure_data.get("documents", {})

        start_time = time.time()

        for file_id, file_info in spd_files.items():
            file_path = file_info.get("path", "")
            title = Path(file_path).name

            # Extract SPD number
            spd_match = self.SPD_PATTERN.search(file_path)
            spd_num = spd_match.group(1) if spd_match else "XX"

            # Get sections
            doc_structure = structures.get(file_id, {})
            sections = doc_structure.get("sections", [])
            section_titles = [s.get("title", "")[:60] for s in sections[:15]]
            sections_text = "\n".join(f"- {t}" for t in section_titles) if section_titles else "Non structuré"

            # Get extracts
            doc_extracts = extracts.get(file_id, {})
            sentences = doc_extracts.get("sentences", [])
            extract_texts = [s.get("text", "")[:300] for s in sentences[:8]]
            extracts_text = "\n".join(f"- {t}" for t in extract_texts) if extract_texts else "Aucun extrait"

            # Build prompt
            prompt = prompt_template.format(
                title=title,
                sections=sections_text,
                extracts=extracts_text
            )

            # Call LLM
            try:
                response = ollama.generate(
                    model=llm_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.2,
                        "num_predict": 800,
                    }
                )
                raw_response = response.get("response", "")

                # Parse functionalities
                funcs = self._parse_functionalities(raw_response, spd_num, file_id, file_path, language)

                for func in funcs:
                    # Classify category
                    func["category"] = self._classify_functionality(func, language)

                    # Extract cross-references
                    refs = func.get("references", [])
                    for ref in refs:
                        if ref != f"SPD-{spd_num}":
                            cross_references[f"SPD-{spd_num}"].append(ref)

                    all_functionalities.append(func)
                    by_document[file_id].append(func)

                logger.debug(f"[doc_func_extract] {title}: {len(funcs)} functionalities")

            except Exception as e:
                logger.warning(f"[doc_func_extract] Error for {file_id}: {e}")
                errors.append({"file_id": file_id, "error": str(e)})

        elapsed = time.time() - start_time

        # Group by category
        by_category: Dict[str, List[Dict]] = defaultdict(list)
        for func in all_functionalities:
            cat = func.get("category", "other")
            by_category[cat].append(func)

        # Build reference validation
        existing_spds = set()
        for fid, finfo in spd_files.items():
            match = self.SPD_PATTERN.search(finfo.get("path", ""))
            if match:
                existing_spds.add(f"SPD-{match.group(1)}")

        missing_references = []
        for source, targets in cross_references.items():
            for target in targets:
                if target not in existing_spds and target.startswith("SPD-"):
                    missing_references.append({
                        "source": source,
                        "target": target,
                        "type": "missing_reference"
                    })

        # Statistics
        statistics = {
            "spd_documents": len(spd_files),
            "total_functionalities": len(all_functionalities),
            "documents_with_funcs": len(by_document),
            "avg_funcs_per_doc": round(len(all_functionalities) / len(spd_files), 1) if spd_files else 0,
            "categories": {k: len(v) for k, v in by_category.items()},
            "cross_references": sum(len(v) for v in cross_references.values()),
            "missing_references": len(missing_references),
            "errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info(
            f"[doc_func_extract] Extracted {len(all_functionalities)} functionalities "
            f"from {len(spd_files)} SPD documents in {elapsed:.1f}s"
        )

        return {
            "functionalities": all_functionalities,
            "by_document": dict(by_document),
            "by_category": dict(by_category),
            "cross_references": dict(cross_references),
            "missing_references": missing_references,
            "spd_inventory": list(existing_spds),
            "errors": errors,
            "statistics": statistics,
        }

    def _parse_functionalities(
        self,
        response: str,
        spd_num: str,
        file_id: str,
        file_path: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured functionalities."""
        functionalities = []

        # Split by separator
        blocks = re.split(r"\n---+\n", response)

        func_counter = 1
        for block in blocks:
            if not block.strip():
                continue

            func = {
                "id": f"SPD-{spd_num}-F{func_counter:02d}",
                "spd_number": spd_num,
                "source_file_id": file_id,
                "source_path": file_path,
                "name": "",
                "description": "",
                "actors": [],
                "trigger": "",
                "references": [],
                "raw_block": block,
            }

            # Parse fields based on language
            if language == "fr":
                patterns = {
                    "id": r"ID\s*:\s*(.+)",
                    "name": r"NOM\s*:\s*(.+)",
                    "description": r"DESCRIPTION\s*:\s*(.+)",
                    "actors": r"ACTEURS?\s*:\s*(.+)",
                    "trigger": r"DÉCLENCHEUR\s*:\s*(.+)",
                    "references": r"RÉFÉRENCES?\s*:\s*(.+)",
                }
            else:
                patterns = {
                    "id": r"ID\s*:\s*(.+)",
                    "name": r"NAME\s*:\s*(.+)",
                    "description": r"DESCRIPTION\s*:\s*(.+)",
                    "actors": r"ACTORS?\s*:\s*(.+)",
                    "trigger": r"TRIGGER\s*:\s*(.+)",
                    "references": r"REFERENCES?\s*:\s*(.+)",
                }

            for field, pattern in patterns.items():
                match = re.search(pattern, block, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()

                    if field == "id" and value:
                        func["id"] = value

                    elif field in ["actors", "references"]:
                        # Parse as list
                        items = re.split(r"[,;]", value)
                        items = [i.strip().strip("[]") for i in items if i.strip()]
                        func[field] = items

                    else:
                        func[field] = value

            # Only add if we have at least a name or description
            if func["name"] or func["description"]:
                functionalities.append(func)
                func_counter += 1

        return functionalities

    def _classify_functionality(self, func: Dict[str, Any], language: str) -> str:
        """Classify functionality into a category."""
        text = f"{func.get('name', '')} {func.get('description', '')}".lower()

        for category, keywords in self.CATEGORIES.items():
            for kw in keywords:
                if kw in text:
                    return category

        return "other"

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total = stats.get("total_functionalities", 0)
        docs = stats.get("spd_documents", 0)
        avg = stats.get("avg_funcs_per_doc", 0)
        missing = stats.get("missing_references", 0)

        return (
            f"Functionalities: {total} extracted from {docs} SPD documents "
            f"(avg {avg}/doc). {missing} missing references detected."
        )
