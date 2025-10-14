"""Pipeline utilities for the Hackaging theories challenge."""

from .literature import (
    LiteratureRetriever,
    PaperMetadata,
    PaperSection,
    ProviderConfig,
    RetrievalResult,
)
from .ontology import TheoryOntology
from .filtering import RelevanceFilter, FilterDecision
from .review_bootstrap import bootstrap_ontology, BootstrapSummary
from .theories import TheoryClassifier, TheoryAssignment
from .extraction import QuestionExtractor, QuestionAnswer
from .outputs import (
    export_papers,
    export_theories,
    export_question_answers,
)
from .pipeline_utils import classify_and_extract_parallel
from .llm import LLMClient, LLMClientConfig, LLMClientError, LLMRateLimitError

__all__ = [
    "LiteratureRetriever",
    "PaperMetadata",
    "PaperSection",
    "ProviderConfig",
    "RetrievalResult",
    "TheoryClassifier",
    "TheoryAssignment",
    "TheoryOntology",
    "RelevanceFilter",
    "FilterDecision",
    "QuestionExtractor",
    "QuestionAnswer",
    "bootstrap_ontology",
    "BootstrapSummary",
    "classify_and_extract_parallel",
    "export_papers",
    "export_theories",
    "export_question_answers",
    "LLMClient",
    "LLMClientConfig",
    "LLMClientError",
    "LLMRateLimitError",
]
