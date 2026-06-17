"""Test each model individually on one Easy problem to identify timeouts."""
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.config import load_config
from src.providers import create_provider
from src.dataset import load_leetcode_dataset
from src.agent import Agent, generate_initial_solution

load_dotenv()

config = load_config("configs/baseline.yaml")
probs = load_leetcode_dataset(10, 99)
problem = next(p for p in probs if p['difficulty'] == 'Hard')
print(f"Problem: {problem['problem_id']} ({problem['difficulty']})\n")

for i, pcfg in enumerate(config.providers):
    name = f"[{i+1}] {pcfg.name}"
    provider = create_provider(pcfg)
    agent = Agent(id=i, provider=provider)
    start = time.time()
    try:
        sol, prompt, resp = generate_initial_solution(agent, problem)
        elapsed = time.time() - start
        print(f"  ✅ {name} — {elapsed:.1f}s — code {len(sol.code)} chars")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ {name} — {elapsed:.1f}s — {type(e).__name__}: {e}")

print("\nDone.")
