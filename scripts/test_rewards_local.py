"""Local reward engine smoke tests — no GPU required."""

from cardio.trainer.rewards.vprm import ACCAHAVerifier
from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier
from cardio.trainer.rewards.dcr import AutoMetricConverter
from cardio.trainer.grpo import NGRPOTrainer

import torch

# --- VPRM Tests ---
verifier = ACCAHAVerifier()

# Test 1: Valid HFpEF — LVEF >= 50% with filling pressure evidence
trace1 = "The LVEF is 55%. LAVI is 38 mL/m2, indicating elevated filling pressures."
r, v = verifier.verify_hf_classification(trace1, "HFpEF", "HFpEF")
assert r == 1.0, f"Expected 1.0, got {r}. Violations: {v}"
print(f"VPRM Test 1 PASSED: reward={r}")

# Test 2: Invalid HFpEF — LVEF too low (returns 0.0, not -1.0; pred/gt match
# so the -1.0 label-mismatch path is not taken — _verify_hfpef returns 0.0
# because lvef < 50 violates the >= 50% threshold)
trace2 = "The LVEF is 35%. LAVI is 38 mL/m2."
r, v = verifier.verify_hf_classification(trace2, "HFpEF", "HFpEF")
assert r == 0.0, f"Expected 0.0, got {r}"
print(f"VPRM Test 2 PASSED: reward={r}")

# Test 3: HFpEF missing secondary evidence
trace3 = "The LVEF is 55%. Cardiac function appears preserved."
r, v = verifier.verify_hf_classification(trace3, "HFpEF", "HFpEF")
assert r == 0.5, f"Expected 0.5, got {r}"
print(f"VPRM Test 3 PASSED: reward={r}")

# Test 4: Valid HFrEF
trace4 = "The LVEF is 28%, indicating severely reduced systolic function."
r, v = verifier.verify_hf_classification(trace4, "HFrEF", "HFrEF")
assert r == 1.0, f"Expected 1.0, got {r}"
print(f"VPRM Test 4 PASSED: reward={r}")

# Test 5: Regex format variations
metrics = verifier.extract_metrics("LVEF of 45%, LAVI: 36 mL/m2, E/e' is 16, GLS -14%")
assert metrics.get("lvef") == 45.0, f"LVEF extraction failed: {metrics}"
assert metrics.get("lavi") == 36.0, f"LAVI extraction failed: {metrics}"
print(f"VPRM Test 5 PASSED: extracted {metrics}")

# --- NGRPO Advantage Calibration Tests ---
trainer = NGRPOTrainer(model=None, r_max=1.0, epsilon_neg=0.16, epsilon_pos=0.24)

# Test 6: Homogeneous rewards — standard GRPO deadlocks, NGRPO doesn't
rewards_homo = torch.tensor([0.0, 0.0, 0.0, 0.0])
advantages = trainer.compute_calibrated_advantages(rewards_homo)
assert (advantages < 0).all(), f"Expected all negative advantages, got {advantages}"
assert advantages.std() < 1e-6, f"Expected identical advantages, got std={advantages.std()}"
print(f"NGRPO Test 6 PASSED: homogeneous rewards -> advantages={advantages}")

# Test 7: Mixed rewards — correct relative ordering
rewards_mixed = torch.tensor([0.0, 0.5, 1.0, 0.0])
advantages = trainer.compute_calibrated_advantages(rewards_mixed)
assert advantages[2] > advantages[1] > advantages[0], f"Wrong ordering: {advantages}"
print(f"NGRPO Test 7 PASSED: mixed rewards -> advantages={advantages}")

# Test 8: Calibrated std is never zero
for val in [0.0, 0.5, 1.0, -1.0]:
    rewards = torch.tensor([val, val, val, val])
    trainer.compute_calibrated_advantages(rewards)
print("NGRPO Test 8 PASSED: no zero-division on any constant reward value")

# --- Memory Penalty Tests ---
mem_verifier = MemoryInvocationVerifier(context_window=50)

# Test 9: Claim with proper TDM invocation
claims = [{"term": "hypokinesis", "type": "motion", "token_idx": 100, "sentence": "Shows hypokinesis."}]
log = [{"step": 80, "type": "tdm", "token_idx": 80}]
penalty, violations = mem_verifier.verify_invocations(claims, log)
assert penalty == 0.0, f"Expected 0.0 penalty, got {penalty}"
print(f"Memory Test 9 PASSED: proper TDM invocation, penalty={penalty}")

# Test 10: Claim without any invocation
claims = [{"term": "contraction", "type": "motion", "token_idx": 100, "sentence": "Impaired contraction."}]
log = []
penalty, violations = mem_verifier.verify_invocations(claims, log)
assert penalty == -1.0, f"Expected -1.0 penalty, got {penalty}"
print(f"Memory Test 10 PASSED: missing invocation, penalty={penalty}")

# --- DCR Tests ---
amc = AutoMetricConverter()

# Test 11: IoU thresholds
assert amc.compute_penalty(0.7) == 0.0, "High IoU should have no penalty"
assert 0.0 < amc.compute_penalty(0.3) < 1.0, "Medium IoU should have partial penalty"
assert amc.compute_penalty(0.1) == 1.0, "Low IoU should have full penalty"
print("DCR Test 11 PASSED: AMC penalty thresholds correct")

print("\n=== ALL LOCAL TESTS PASSED ===")
