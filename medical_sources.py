"""Comprehensive WHO and Bulgarian medical source catalog for RoseMed."""

from __future__ import annotations

from typing import Dict, List

# Auto-built from WHO fact-sheet slugs (80+ conditions)
WHO_FACT_SHEET_SLUGS: List[str] = [
    "diabetes", "hypertension", "cancer", "stroke", "epilepsy", "depression",
    "malaria", "tuberculosis", "hiv-aids", "obesity-and-overweight",
    "antimicrobial-resistance", "immunization-coverage", "asthma",
    "chronic-obstructive-pulmonary-disease-copd", "dementia", "diarrhoeal-disease",
    "hepatitis-b", "hepatitis-c", "hepatitis-e", "influenza-seasonal",
    "measles", "meningococcal-meningitis", "mpox", "pneumococcal-diseases",
    "poliomyelitis", "rabies", "tetanus", "typhoid", "cholera", "dengue-and-severe-dengue",
    "ebola-virus-disease", "zika-virus", "yellow-fever", "chikungunya",
    "leishmaniasis", "onchocerciasis", "schistosomiasis", "trypanosomiasis-human-african",
    "lymphatic-filariasis", "soil-transmitted-helminth-infections", "taeniasis-cysticercosis",
    "covid-19", "influenza-avian", "respiratory-syncytial-virus",
    "cardiovascular-diseases-(cvds)", "coronary-heart-disease", "rheumatic-heart-disease",
    "chronic-kidney-disease", "diabetes-blindness", "glaucoma", "deafness-and-hearing-loss",
    "autism-spectrum-disorders", "adhd", "schizophrenia", "suicide",
    "substance-use-disorders", "alcohol", "tobacco", "cannabis",
    "child-maltreatment", "elder-abuse", "female-genital-mutilation",
    "infant-and-young-child-feeding", "malnutrition", "anaemia",
    "food-safety", "physical-activity", "healthy-diet", "road-traffic-injuries",
    "falls", "burns", "drowning", "lead-poisoning-and-health",
    "asbestos", "air-pollution-and-health", "climate-change-and-health",
    "cervical-cancer", "breast-cancer", "colorectal-cancer", "prostate-cancer",
    "palliative-care", "pain-management", "blood-safety", "organ-donation",
    "health-systems", "universal-health-coverage", "health-workforce",
    "essential-medicines", "rational-use-of-medicines", "pharmacovigilance",
    "maternal-mortality", "newborn-mortality", "preterm-birth", "stillbirth",
    "contraception", "sexually-transmitted-infections", "infertility",
    "violence-against-women", "child-health", "adolescent-health",
    "ageing-and-health", "disability-and-health", "rare-diseases",
    "genomics-and-health", "antibiotic-resistance", "sepsis",
    "surgical-site-infections", "health-care-associated-infections",
    "diabetes-foot-care", "sickle-cell-disease", "thalassaemia",
    "haemophilia", "cystic-fibrosis", "down-syndrome", "autism",
    "multiple-sclerosis", "parkinson-disease", "alzheimer-disease",
    "osteoporosis", "arthritis", "lupus", "psoriasis",
    "inflammatory-bowel-disease", "celiac-disease", "food-allergy",
    "anaphylaxis", "snakebite-envenoming", "scorpion-stings",
]

WHO_HEALTH_TOPICS: List[str] = [
    "cardiovascular-diseases", "diabetes", "cancer", "mental-health",
    "neurological-disorders", "respiratory-diseases", "kidney-diseases",
    "liver-diseases", "blood-disorders", "hiv-aids", "tuberculosis",
    "malaria", "neglected-tropical-diseases", "antimicrobial-resistance",
    "vaccines-and-immunization", "maternal-health", "child-health",
    "adolescent-health", "ageing", "disability", "nutrition",
    "obesity", "physical-activity", "substance-use", "tobacco",
    "health-systems", "health-technologies", "medicines",
    "patient-safety", "surgery", "emergencies", "pandemic-preparedness",
    "environmental-health", "occupational-health", "food-safety",
    "water-sanitation-and-hygiene", "injuries-and-violence",
    "road-safety", "rare-diseases", "genomics", "traditional-medicine",
]


def _who_fact_sheet_url(slug: str) -> str:
    return f"https://www.who.int/news-room/fact-sheets/detail/{slug}"


def _who_topic_url(topic: str) -> str:
    return f"https://www.who.int/health-topics/{topic}"


def build_who_sources() -> List[Dict[str, str]]:
    """Build full WHO source list from slugs and health topics."""
    sources: List[Dict[str, str]] = []
    seen: set[str] = set()

    for slug in WHO_FACT_SHEET_SLUGS:
        url = _who_fact_sheet_url(slug)
        if url in seen:
            continue
        seen.add(url)
        title = slug.replace("-", " ").title()
        sources.append({
            "id": f"who_fs_{slug[:40]}",
            "title": f"WHO Fact Sheet: {title}",
            "url": url,
            "org": "WHO",
            "lang": "en",
        })

    for topic in WHO_HEALTH_TOPICS:
        url = _who_topic_url(topic)
        if url in seen:
            continue
        seen.add(url)
        title = topic.replace("-", " ").title()
        sources.append({
            "id": f"who_topic_{topic[:40]}",
            "title": f"WHO Health Topic: {title}",
            "url": url,
            "org": "WHO",
            "lang": "en",
        })

    return sources


WHO_URL_SOURCES: List[Dict[str, str]] = build_who_sources()

# Bulgarian — expanded working endpoints
BULGARIAN_URL_SOURCES: List[Dict[str, str]] = [
    {"id": "nhif_home", "title": "NHIF Bulgaria Home", "url": "https://www.nhif.bg/bg/", "org": "NHIF", "lang": "bg"},
    {"id": "nhif_patients", "title": "NHIF Patient Rights", "url": "https://www.nhif.bg/bg/patient_rights", "org": "NHIF", "lang": "bg"},
    {"id": "nhif_doctors", "title": "NHIF for Doctors", "url": "https://www.nhif.bg/bg/doctors", "org": "NHIF", "lang": "bg"},
    {"id": "nhif_medicines", "title": "NHIF Medicines", "url": "https://www.nhif.bg/bg/medicines", "org": "NHIF", "lang": "bg"},
    {"id": "nhif_hospitals", "title": "NHIF Hospitals", "url": "https://www.nhif.bg/bg/hospitals", "org": "NHIF", "lang": "bg"},
    {"id": "bda_home", "title": "Bulgarian Drug Agency", "url": "https://www.bda.bg/", "org": "BDA", "lang": "bg"},
    {"id": "bda_medicines", "title": "BDA Medicinal Products", "url": "https://www.bda.bg/en/search-medicinal-products", "org": "BDA", "lang": "bg"},
    {"id": "moh_home", "title": "Ministry of Health Bulgaria", "url": "https://www.mh.government.bg/", "org": "MoH Bulgaria", "lang": "bg"},
    {"id": "moh_news", "title": "MoH News", "url": "https://www.mh.government.bg/bg/news/", "org": "MoH Bulgaria", "lang": "bg"},
    {"id": "moh_policies", "title": "MoH Policies", "url": "https://www.mh.government.bg/bg/politiki/", "org": "MoH Bulgaria", "lang": "bg"},
    {"id": "ncphp", "title": "National Center Public Health", "url": "https://ncphp.government.bg/", "org": "NCPHP Bulgaria", "lang": "bg"},
    {"id": "emergency_bg", "title": "Emergency Medical Care Bulgaria", "url": "https://www.mh.government.bg/bg/deynosti/speshna-pomosht/", "org": "MoH Bulgaria", "lang": "bg"},
    {"id": "immunization_bg", "title": "Bulgarian Immunization Program", "url": "https://www.mh.government.bg/bg/deynosti/imunizacii/", "org": "MoH Bulgaria", "lang": "bg"},
    {"id": "euro_who_bg", "title": "WHO Europe Bulgaria Profile", "url": "https://www.who.int/europe/countries/bgr", "org": "WHO Europe", "lang": "en"},
]

# Bulk download URLs (JSON/XML — full datasets)
BULK_DOWNLOAD_SOURCES: List[Dict[str, str]] = [
    {
        "id": "orphanet_rare_diseases",
        "title": "Orphanet Rare Disease Classification",
        "url": "https://www.orpha.net/data/xml/en_product1.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "orphanet_linearisation",
        "title": "Orphanet Linearisation",
        "url": "https://www.orpha.net/data/xml/en_product6.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "orphanet_epidemiology",
        "title": "Orphanet Epidemiology",
        "url": "https://www.orpha.net/data/xml/en_product2.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "orphanet_natural_history",
        "title": "Orphanet Natural History",
        "url": "https://www.orpha.net/data/xml/en_product3.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "orphanet_management",
        "title": "Orphanet Management and Treatment",
        "url": "https://www.orpha.net/data/xml/en_product4.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "orphanet_clinical_signs",
        "title": "Orphanet Clinical Signs",
        "url": "https://www.orpha.net/data/xml/en_product9.xml",
        "org": "Orphanet",
        "lang": "en",
        "type": "xml",
    },
    {
        "id": "who_essential_medicines",
        "title": "WHO Essential Medicines List",
        "url": "https://list.essentialmeds.org/",
        "org": "WHO",
        "lang": "en",
        "type": "html",
    },
]

RAW_SOURCES_DIR_NAME = "raw"
BULK_CACHE_DIR_NAME = "bulk"

SUPPORTED_RAW_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".pdf", ".csv", ".xml"}
