"""Training entry points (placeholders — nothing is trained yet).

Each stage exposes ``train(config)``. PyTorch is imported lazily inside the
implementations once they exist, so the package stays importable without it.
"""
