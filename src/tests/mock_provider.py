"""Mock LLM provider for testing debate engine without API calls."""

from src.providers.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Programmable mock provider for testing.

    Returns predetermined responses based on call type and agent id.
    """

    def __init__(self, name: str = "mock", model: str = "mock-model"):
        self._name = name
        self._model = model
        # Programmable responses: call_count → response
        self.generate_responses: list[str] = []
        self.refine_responses: list[str] = []
        self.vote_responses: list[str] = []
        self._call_history: list[dict] = []
        self._gen_idx = 0
        self._refine_idx = 0
        self._vote_idx = 0

    @property
    def model_name(self) -> str:
        return f"mock:{self._model}"

    @property
    def call_history(self) -> list[dict]:
        return self._call_history

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self._call_history.append({
            "system": system_prompt[:80],
            "user": user_prompt[:80],
        })

        # Determine call type from system prompt content
        if "generate" in system_prompt.lower() or "expert Python" in system_prompt.lower():
            if self._gen_idx < len(self.generate_responses):
                resp = self.generate_responses[self._gen_idx]
                self._gen_idx += 1
                return resp
            return self._default_code()
        elif "refine" in system_prompt.lower() or "code reviewer" in system_prompt.lower():
            if self._refine_idx < len(self.refine_responses):
                resp = self.refine_responses[self._refine_idx]
                self._refine_idx += 1
                return resp
            return self._default_code()
        elif "judge" in system_prompt.lower() or "rank" in system_prompt.lower():
            if self._vote_idx < len(self.vote_responses):
                resp = self.vote_responses[self._vote_idx]
                self._vote_idx += 1
                return resp
            return "[1]"
        else:
            return self._default_code()

    def _default_code(self) -> str:
        return (
            "class Solution:\n"
            "    def solve(self, nums, target):\n"
            "        return [0, 1]\n"
        )

    def reset(self):
        self._gen_idx = 0
        self._refine_idx = 0
        self._vote_idx = 0
        self._call_history.clear()


# ---------------------------------------------------------------------------
# Pre-built test fixtures
# ---------------------------------------------------------------------------

def make_mock_agent(
    agent_id: int,
    gen_code: str | None = None,
    vote_seq: str | None = None,
) -> tuple:
    """Create an Agent with MockProvider for testing.

    Returns (Agent, MockProvider) so tests can inspect call history.
    """
    provider = MockProvider(name=f"agent-{agent_id}")
    if gen_code:
        provider.generate_responses.append(gen_code)
    if vote_seq:
        provider.vote_responses.append(vote_seq)

    from src.agent import Agent
    agent = Agent(id=agent_id, provider=provider)
    return agent, provider


SAMPLE_CODE_A = (
    "class Solution:\n"
    "    def solve(self, nums, target):\n"
    "        d = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in d:\n"
    "                return [d[target - x], i]\n"
    "            d[x] = i\n"
)

SAMPLE_CODE_B = (
    "class Solution:\n"
    "    def solve(self, nums, target):\n"
    "        for i in range(len(nums)):\n"
    "            for j in range(i+1, len(nums)):\n"
    "                if nums[i] + nums[j] == target:\n"
    "                    return [i, j]\n"
)

SAMPLE_CODE_C = (
    "class Solution:\n"
    "    def solve(self, nums, target):\n"
    "        nums.sort()\n"
    "        l, r = 0, len(nums)-1\n"
    "        while l < r:\n"
    "            s = nums[l] + nums[r]\n"
    "            if s == target:\n"
    "                return [l, r]\n"
    "            elif s < target:\n"
    "                l += 1\n"
    "            else:\n"
    "                r -= 1\n"
)

SAMPLE_CODES = [SAMPLE_CODE_A, SAMPLE_CODE_B, SAMPLE_CODE_C]


def make_test_problem() -> dict:
    """Minimal problem dict for testing."""
    return {
        "title": "Two Sum",
        "difficulty": "Easy",
        "description": "Find two numbers that add up to target.",
        "constraints": "2 <= len(nums) <= 10^4",
        "function_signature": "solve(self, nums: List[int], target: int) -> List[int]",
        "starter_code": "class Solution:\n    def solve(self, nums, target):\n        pass\n",
        "entry_point": "Solution().solve",
        "test": "def check(candidate):\n    assert candidate(nums=[2,7,11,15],target=9) == [0,1]\n",
    }
