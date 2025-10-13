"""Pipeline utilities for the Hackaging theories challenge."""

from .literature import (
    LiteratureRetriever,
    PaperMetadata,
    PaperSection,
    ProviderConfig,
    RetrievalResult,
)
from .ontology import TheoryOntology
from .ontology_manager import OntologyManager, OntologyUpdate
from .theories import TheoryClassifier, TheoryAssignment
from .extraction import QuestionExtractor, QuestionAnswer
from .outputs import (
    export_papers,
    export_theories,
    export_question_answers,
)
from .pipeline_utils import classify_and_extract_parallel
from .llm import LLMClient, LLMClientConfig, LLMClientError, LLMRateLimitError
from .review_bootstrap import (
    BootstrapResult,
    ReviewDocument,
    build_bootstrap_ontology,
    extract_theories_from_review,
    merge_bootstrap_into_targets,
    normalise_review_metadata,
    pull_top_cited_reviews,
    write_bootstrap_cache,
)

__all__ = [
    "LiteratureRetriever",
    "PaperMetadata",
    "PaperSection",
    "ProviderConfig",
    "RetrievalResult",
    "TheoryClassifier",
    "TheoryAssignment",
    "TheoryOntology",
    "OntologyManager",
    "OntologyUpdate",
    "QuestionExtractor",
    "QuestionAnswer",
    "classify_and_extract_parallel",
    "export_papers",
    "export_theories",
    "export_question_answers",
    "LLMClient",
    "LLMClientConfig",
    "LLMClientError",
    "LLMRateLimitError",
    "ReviewDocument",
    "BootstrapResult",
    "pull_top_cited_reviews",
    "normalise_review_metadata",
    "extract_theories_from_review",
    "build_bootstrap_ontology",
    "merge_bootstrap_into_targets",
    "write_bootstrap_cache",
]
