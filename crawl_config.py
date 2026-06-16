"""Medical web crawl configuration — whitelisted domains only."""

from __future__ import annotations

from typing import Dict, List, Set

# Domains we are allowed to crawl (medical / public health only)
CRAWL_ALLOWED_DOMAINS: Set[str] = {
    "who.int",
    "www.who.int",
    "orpha.net",
    "www.orpha.net",
    "orphadata.com",
    "www.orphadata.com",
    "nhif.bg",
    "www.nhif.bg",
    "bda.bg",
    "www.bda.bg",
    "mh.government.bg",
    "www.mh.government.bg",
    "ncphp.government.bg",
    "www.ncphp.government.bg",
    "ecdc.europa.eu",
    "www.ecdc.europa.eu",
    "ema.europa.eu",
    "www.ema.europa.eu",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "clinicaltrials.gov",
    "www.clinicaltrials.gov",
    "nice.org.uk",
    "www.nice.org.uk",
    "cdc.gov",
    "www.cdc.gov",
    "nih.gov",
    "www.nih.gov",
    "health.gov",
    "www.health.gov",
    "list.essentialmeds.org",
    "www.list.essentialmeds.org",
    "euro.who.int",
    "iris.who.int",
    "apps.who.int",
    "medlineplus.gov",
    "www.medlineplus.gov",
    "open.fda.gov",
    "www.open.fda.gov",
    "disease-ontology.org",
    "www.disease-ontology.org",
}

# BFS seed URLs — crawl expands from here across allowed domains
CRAWL_SEED_URLS: List[str] = [
    "https://www.who.int/health-topics",
    "https://www.who.int/news-room/fact-sheets",
    "https://www.who.int/publications",
    "https://www.who.int/europe/countries/bgr",
    "https://www.who.int/teams/global-programme-on-tuberculosis",
    "https://www.who.int/teams/immunization-vaccines-and-biologicals",
    "https://www.nhif.bg/bg/",
    "https://www.nhif.bg/bg/medicines",
    "https://www.nhif.bg/bg/hospitals",
    "https://www.bda.bg/",
    "https://www.mh.government.bg/",
    "https://www.mh.government.bg/bg/politiki/",
    "https://ncphp.government.bg/",
    "https://www.ecdc.europa.eu/en/health-topics",
    "https://www.ema.europa.eu/en/medicines",
    "https://www.nice.org.uk/guidance/published",
    "https://medlineplus.gov/healthtopics.html",
    "https://list.essentialmeds.org/",
    "https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN",
    "https://clinicaltrials.gov/search?cond=diabetes",
]

# URL path substrings to skip (login, search params noise, media)
CRAWL_SKIP_PATTERNS: List[str] = [
    "/login", "/signin", "/cart", "/checkout", "/account",
    "javascript:", "mailto:", "#", ".jpg", ".png", ".gif", ".zip",
    "/search?", "facebook.com", "twitter.com", "linkedin.com", "youtube.com",
]

# Org label by domain
DOMAIN_ORG: Dict[str, str] = {
    "who.int": "WHO",
    "orpha.net": "Orphanet",
    "orphadata.com": "Orphanet",
    "nhif.bg": "NHIF",
    "bda.bg": "BDA",
    "mh.government.bg": "MoH Bulgaria",
    "ncphp.government.bg": "NCPHP Bulgaria",
    "ecdc.europa.eu": "ECDC",
    "ema.europa.eu": "EMA",
    "ncbi.nlm.nih.gov": "PubMed/NCBI",
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "clinicaltrials.gov": "ClinicalTrials.gov",
    "nice.org.uk": "NICE UK",
    "cdc.gov": "CDC",
    "nih.gov": "NIH",
    "medlineplus.gov": "MedlinePlus",
    "list.essentialmeds.org": "WHO Essential Medicines",
}

DEFAULT_PAGES_PER_RUN = 800
CRAWL_DELAY_SECONDS = 0.35
