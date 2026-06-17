"""Smoke test: verify all configured providers can return a response."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.config import load_config
from src.providers import create_provider

load_dotenv()

config = load_config("configs/baseline.yaml")
print(f"Testing {len(config.providers)} providers...\n")

for i, pcfg in enumerate(config.providers):
    name = f"[{i+1}] {pcfg.name} ({pcfg.model})"
    try:
        provider = create_provider(pcfg)
        resp = provider.generate(
            system_prompt="Reply with exactly: OK",
            user_prompt="Say OK",
        )
        short = resp.strip()[:60]
        print(f"  ✅ {name} → {short}")
    except Exception as e:
        print(f"  ❌ {name} → {type(e).__name__}: {e}")

print("\nDone.")
