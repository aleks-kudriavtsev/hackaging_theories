"""Pipeline utilities for the Hackaging theories challenge."""

from .literature import (
    LiteratureRetriever,
    PaperMetadata,
    PaperSection,
    ProviderConfig,
    RetrievalResult,
)
from .ontology import TheoryOntology
from .ontology_manager import OntologyManager, OntologyUpdate, RuntimeNodeSpec
from .ontology_suggestions import (
    load_ontology_query_suggestions,
    merge_query_suggestions,
)
from .theories import TheoryClassifier, TheoryAssignment
from .extraction import QuestionExtractor, QuestionAnswer
from .outputs import (
    export_papers,
    export_theories,
    export_question_answers,
)
from .pipeline_utils import classify_and_extract_parallel
from .filtering import RelevanceFilter, FilterDecision
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
from .runtime_bootstrap import (
    RuntimeLabelRequest,
    RuntimeLabelResponse,
    RuntimeOntologyBootstrapper,
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
    "RuntimeNodeSpec",
    "load_ontology_query_suggestions",
    "merge_query_suggestions",
    "QuestionExtractor",
    "QuestionAnswer",
    "classify_and_extract_parallel",
    "RelevanceFilter",
    "FilterDecision",
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
    "RuntimeOntologyBootstrapper",
    "RuntimeLabelRequest",
    "RuntimeLabelResponse",
]
