from typing import Any, Mapping
from services.generation.base import BaseGenerationProvider

def build_generation_provider(cfg: Mapping[str, Any]) -> BaseGenerationProvider:
    provider_type = cfg.get("GENERATION_PROVIDER", "local_t5")
    
    if provider_type == "local_t5":
        from services.generation.local_t5_provider import LocalT5Provider
        return LocalT5Provider(cfg)
    elif provider_type == "gemini":
        from services.generation.gemini_provider import GeminiProvider
        return GeminiProvider(cfg)
    else:
        raise ValueError(f"Unknown generation provider type: {provider_type}")
