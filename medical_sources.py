"""Catalog of WHO and Bulgarian medical data sources for RoseMed."""

from __future__ import annotations

from typing import Any, Dict, List

# Public WHO fact sheets and health topics (English — model translates to Bulgarian)
WHO_URL_SOURCES: List[Dict[str, str]] = [
    {
        "id": "who_diabetes",
        "title": "Diabetes",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_hypertension",
        "title": "Hypertension",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_cancer",
        "title": "Cancer",
        "url": "https://www.who.int/news-room/fact-sheets/detail/cancer",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_heart_disease",
        "title": "Cardiovascular diseases",
        "url": "https://www.who.int/health-topics/cardiovascular-diseases",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_stroke",
        "title": "Stroke",
        "url": "https://www.who.int/news-room/fact-sheets/detail/stroke",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_epilepsy",
        "title": "Epilepsy",
        "url": "https://www.who.int/news-room/fact-sheets/detail/epilepsy",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_depression",
        "title": "Depression",
        "url": "https://www.who.int/news-room/fact-sheets/detail/depression",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_antimicrobial_resistance",
        "title": "Antimicrobial resistance",
        "url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_immunization",
        "title": "Immunization coverage",
        "url": "https://www.who.int/news-room/fact-sheets/detail/immunization-coverage",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_malaria",
        "title": "Malaria",
        "url": "https://www.who.int/news-room/fact-sheets/detail/malaria",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_tb",
        "title": "Tuberculosis",
        "url": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_hiv",
        "title": "HIV/AIDS",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_obesity",
        "title": "Obesity and overweight",
        "url": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_mental_health",
        "title": "Mental health",
        "url": "https://www.who.int/health-topics/mental-health",
        "org": "WHO",
        "lang": "en",
    },
    {
        "id": "who_rare_diseases",
        "title": "Genomics and rare diseases",
        "url": "https://www.who.int/health-topics/genomics-and-rare-diseases",
        "org": "WHO",
        "lang": "en",
    },
]

# Bulgarian public health portals — fetch where possible; drop PDFs/HTML in data/sources/raw/
BULGARIAN_URL_SOURCES: List[Dict[str, str]] = [
    {
        "id": "moh_bg_health",
        "title": "Ministry of Health Bulgaria",
        "url": "https://www.mh.government.bg/bg/health/",
        "org": "MoH Bulgaria",
        "lang": "bg",
    },
    {
        "id": "nhif_bg",
        "title": "NHIF Bulgaria",
        "url": "https://www.nhif.bg/bg/",
        "org": "NHIF",
        "lang": "bg",
    },
    {
        "id": "bda_bg",
        "title": "Bulgarian Drug Agency",
        "url": "https://www.bda.bg/",
        "org": "BDA",
        "lang": "bg",
    },
]

# Drop local files here: WHO PDFs, NZOK lists, MoH guidelines, hospital SOPs
RAW_SOURCES_DIR_NAME = "raw"

SUPPORTED_RAW_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".pdf", ".csv"}
