"""
EXP 00 — Infrastructure Validation
Verify Qwen 2.5 3B loads, hooks work, and hidden states have variance.
Run this before booking any GPU time.
"""
import sys
import numpy as np

print("=" * 60)
print("FAIL-CHAIN — EXP 00: Infrastructure Validation")
print("=" * 60)


def check_existing_artifacts():
    """Check whether HaRP/GEOM-PROOF hidden states can be reused."""
    import os
    candidates = [
        "../harp/results/hidden_states/hidden_states.npz",
        "../geom-proof/results/hidden_states/qwen_hidden_states.npz",
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path)
            hs = data["hidden_states"]
            print(f"[FOUND] {path}: shape={hs.shape}")
            if hs.shape[1] >= 33:  # need at least L32
                print(f"  → L32 available. Reusable for EXP 02 Fisher trajectory.")
                return True, path
    print("[NOT FOUND] No existing hidden states. EXP 01 will generate fresh.")
    return False, None


def gate1_model_loads():
    """Gate 1: Qwen 2.5 3B loads without error."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("\nGate 1: Loading Qwen 2.5 3B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        print("  → Tokenizer loaded.")

        # Test tokenization
        test_ids = tokenizer("What is the capital of France?", return_tensors="pt")
        assert test_ids["input_ids"].shape[1] > 0
        print("  → Tokenization works.")
        print("Gate 1 PASSED\n")
        return True
    except Exception as e:
        print(f"Gate 1 FAILED: {e}")
        print("  → Check: HuggingFace cache, model name, internet connection")
        return False


def gate2_hook_registers():
    """Gate 2: Hook on L32 captures correct shape."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Gate 2: Registering hook on L32...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        captured = {}

        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

        handle = model.model.layers[32].register_forward_hook(hook)

        inputs = tokenizer("What is the capital of France?",
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        handle.remove()

        assert "h" in captured, "Hook did not fire"
        expected_dim = model.config.hidden_size
        assert captured["h"].shape == (expected_dim,), \
            f"Shape mismatch: got {captured['h'].shape}, expected ({expected_dim},)"

        print(f"  → Hidden state shape: {captured['h'].shape} ✓")
        print(f"  → L2 norm: {np.linalg.norm(captured['h']):.3f}")
        print("Gate 2 PASSED\n")
        del model
        return True

    except Exception as e:
        print(f"Gate 2 FAILED: {e}")
        return False


def gate3_variance_check():
    """Gate 3: Hidden states have non-trivial variance across 10 samples."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Gate 3: Checking hidden state variance across 10 samples...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        test_questions = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What is 2 + 2?",
            "When did World War 2 end?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
            "What is the boiling point of water?",
            "Who invented the telephone?",
            "What is the largest planet?",
            "When was the Eiffel Tower built?",
        ]

        captured = {}

        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

        handle = model.model.layers[32].register_forward_hook(hook)

        hidden_states = []
        for q in test_questions:
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            hidden_states.append(captured["h"].copy())

        handle.remove()

        hs_array = np.array(hidden_states)
        mean_std = hs_array.std(0).mean()

        assert mean_std > 0.01, \
            f"Gate 3 FAILED: mean std={mean_std:.4f} < 0.01 — all hidden states look identical"

        print(f"  → Mean std across 10 samples: {mean_std:.4f}")
        print(f"  → Min std: {hs_array.std(0).min():.4f}")
        print(f"  → Max std: {hs_array.std(0).max():.4f}")
        print("Gate 3 PASSED\n")
        del model
        return True

    except Exception as e:
        print(f"Gate 3 FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n[Step 0] Checking for reusable artifacts...")
    found, path = check_existing_artifacts()

    print("\n[Step 1] Running infrastructure gates...")
    print("NOTE: Gates 2 and 3 require GPU. Skip on CPU-only machines.")
    print("      Run on Kaggle before booking full GPU session.\n")

    g1 = gate1_model_loads()

    if "--full" in sys.argv:
        g2 = gate2_hook_registers()
        g3 = gate3_variance_check()
        all_pass = g1 and g2 and g3
    else:
        print("Skipping Gates 2+3 (pass --full to run on GPU)")
        all_pass = g1

    print("=" * 60)
    if all_pass:
        print("ALL GATES PASSED — Ready to run EXP 01 on Kaggle GPU")
    else:
        print("GATES FAILED — Resolve issues before booking GPU time")
    print("=" * 60)
