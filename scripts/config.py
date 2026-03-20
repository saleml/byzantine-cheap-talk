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

# =====================================================================
# Model cohorts
# =====================================================================

AGENTS_V1 = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
    {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
]

AGENTS_V2 = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "gpt-4o", "model_family": "GPT-4o"},
    {"name": "Agent_4", "model": "claude-sonnet-4-6", "model_family": "Claude Sonnet"},
]

AGENTS_V3 = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "gpt-4o", "model_family": "GPT-4o"},
    {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
]

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
