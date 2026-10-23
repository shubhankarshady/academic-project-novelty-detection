"""
Explainable AI (XAI) Module for Academic Project Novelty Detection.
Provides section-level similarity ranking, TF-IDF keyword evidence,
and additive Ridge model contribution attribution.
"""

from .explainer import NoveltyExplainer

__all__ = ["NoveltyExplainer"]
