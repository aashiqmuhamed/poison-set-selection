"""SAILS on the WebShop agentic backdoor (paper §F.5).

Multi-turn trajectory poisoning of a Qwen3-4B WebShop agent: force the agent
to silently purchase a specific high-priced sneaker on any sneaker query.
The cheap oracle is first-action ASR (greedy decode + target substring); the
scorer is ModernBERT-base over [SEP]-concatenated trajectory pairs.

See the directory's README.md for the bring-your-own-upstream workflow with
lancopku/agent-backdoor-attacks + LlamaFactory.
"""
