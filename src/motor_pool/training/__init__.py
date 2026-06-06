"""Config-driven QLoRA training (Unsloth on PEFT/bitsandbytes/TRL).

4080-safe defaults live in configs/training.yaml and config.TrainingConfig. The
trainer keeps manual facts out of weights by construction: every example embeds
its retrieved chunks in-context and loss is computed on the response only.
"""
