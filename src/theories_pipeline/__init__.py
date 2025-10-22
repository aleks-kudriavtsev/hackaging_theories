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
from .theories import (
    AggregatedTheory,
    TheoryAggregationResult,
    TheoryAssignment,
    TheoryClassifier,
    aggregate_theory_assignments,
)
from .extraction import QuestionExtractor, QuestionAnswer
from .outputs import (
    COMPETITION_PAPER_COLUMNS,
    COMPETITION_QUESTION_COLUMNS,
    COMPETITION_THEORY_COLUMNS,
    COMPETITION_THEORY_PAPER_COLUMNS,
    QUESTION_COLUMNS,
    QUESTION_CONFIDENCE_COLUMNS,
    export_competition_papers,
    export_competition_question_answers,
    export_competition_theories,
    export_competition_theory_papers,
    export_papers,
    export_question_answers,
    export_theories,
    export_theory_papers,
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
    "AggregatedTheory",
    "TheoryAggregationResult",
    "TheoryClassifier",
    "TheoryAssignment",
    "aggregate_theory_assignments",
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
    "QUESTION_COLUMNS",
    "QUESTION_CONFIDENCE_COLUMNS",
    "COMPETITION_PAPER_COLUMNS",
    "COMPETITION_THEORY_COLUMNS",
    "COMPETITION_THEORY_PAPER_COLUMNS",
    "COMPETITION_QUESTION_COLUMNS",
    "export_papers",
    "export_theory_papers",
    "export_theories",
    "export_question_answers",
    "export_competition_papers",
    "export_competition_theories",
    "export_competition_theory_papers",
    "export_competition_question_answers",
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
