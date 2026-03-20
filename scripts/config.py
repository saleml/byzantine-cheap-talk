"""
Centralized configuration for all experiment scripts.

Defines model cohorts, FD/PC families, and result paths per version.

Version history:
  v1: Original EACL cohort (Mixtral, Qwen, Llama, DeepSeek)
      - Used for: all results reported in the NETYS paper
  v2: Replaced Llama+DeepSeek with GPT-4o+Claude Sonnet
      - Claude Sonnet had severe schema bugs (73-97% malformed outputs)
      - v2_lowercase: same models, communication words lowercased in prompt
  v3: Replaced Claude Sonnet with DeepSeek (kept GPT-4o)
      - Current model cohort for ongoing experiments
"""

from copy import deepcopy

# =====================================================================
# Model cohorts
# =====================================================================

MASTER_AGENT_POOL = [
    {"model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
    {"model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
    {"model": "gpt-4o", "model_family": "GPT-4o"},
    {"model": "claude-sonnet-4-6", "model_family": "Claude Sonnet"},
]

_MASTER_POOL_BY_FAMILY = {a["model_family"]: a for a in MASTER_AGENT_POOL}


def _with_agent_names(pool):
    return [
        {
            "name": f"Agent_{i}",
            "model": agent["model"],
            "model_family": agent["model_family"],
        }
        for i, agent in enumerate(pool, start=1)
    ]


def get_first_n_agents(n: int):
    """Return the first n agents from MASTER_AGENT_POOL with Agent_1..Agent_n names."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(MASTER_AGENT_POOL):
        raise ValueError(f"n={n} exceeds MASTER_AGENT_POOL size={len(MASTER_AGENT_POOL)}")
    return _with_agent_names(MASTER_AGENT_POOL[:n])


def _agents_from_families(families):
    missing = [f for f in families if f not in _MASTER_POOL_BY_FAMILY]
    if missing:
        raise ValueError(f"Unknown model family/families: {missing}")
    return _with_agent_names([deepcopy(_MASTER_POOL_BY_FAMILY[f]) for f in families])


AGENTS_V1 = _agents_from_families(["Mixtral", "Qwen", "Llama", "DeepSeek"])
AGENTS_V2 = _agents_from_families(["Mixtral", "Qwen", "GPT-4o", "Claude Sonnet"])
AGENTS_V3 = _agents_from_families(["Mixtral", "Qwen", "GPT-4o", "DeepSeek"])

AGENTS_BY_VERSION = {"v1": AGENTS_V1, "v2": AGENTS_V2, "v3": AGENTS_V3}

# =====================================================================
# Model family lists (per version)
# =====================================================================

ALL_FAMILIES_BY_VERSION = {
    "v1": ["Mixtral", "Qwen", "Llama", "DeepSeek"],
    "v2": ["Mixtral", "Qwen", "GPT-4o", "Claude Sonnet"],
    "v3": ["Mixtral", "Qwen", "GPT-4o", "DeepSeek"],
}

# NOTE: FD/PC (fast defector / persistent cooperator) classification
# is NOT hardcoded here. It is derived empirically by analyze_results.py
# from the archetype analysis of k=1 Byzantine data. A model family is
# classified as FD if >50% of its honest instances permanently switch
# to Hunt Hare after the first betrayal, and PC otherwise.
# See analyze_results.py:classify_families_from_data().

# =====================================================================
# Result directory naming
# =====================================================================

def results_dir(experiment: str, version: str) -> str:
    """Return the results subdirectory name for an experiment+version.

    Examples:
        results_dir("byzantine", "v1") -> "results/byzantine_v1"
        results_dir("byzantine_soft", "v3") -> "results/byzantine_soft_v3"
    """
    return f"results/{experiment}_{version}"


# =====================================================================
# Helpers
# =====================================================================

def get_agents(version: str):
    if version not in AGENTS_BY_VERSION:
        raise ValueError(f"Unknown version: {version}. Use one of: {list(AGENTS_BY_VERSION)}")
    return AGENTS_BY_VERSION[version]

def get_agent_family_map(version: str):
    return {a["name"]: a["model_family"] for a in get_agents(version)}

def get_all_families(version: str):
    return ALL_FAMILIES_BY_VERSION.get(version, ALL_FAMILIES_BY_VERSION["v1"])
