"""
src package for dataset preparation, filtering, translation, and evaluation
for the math tutoring conversational dataset.

Modules:
--------
- build_dp: Builds preference datasets (Dp) from conversational datasets (Dc).
- generate_dc: Generates conversational datasets (Dc) using student/tutor simulations.
- data_download_chunking: Downloads and chunks datasets for processing.
- data_filtering: Filters math problems for grade-level appropriateness.
- data_filtering_2: Alternate filtering with simplified few-shots.
- data_translate: Translates datasets to Spanish and classifies pedagogically.
- translate_lib: Provides translation, classification, and summarization utilities.
- util_translate_summarize: Core logic for translation, classification, summarization.
- llms: Factory for Ollama LLM instances.
- context: Builds conversational context for tutor/student prompts.
- validate: Validates JSON output from models against rubric codes.
- rubrics: Rubric code mappings for evaluation, action, and subproblem state.

Usage:
------
Import the needed functions/classes depending on the pipeline stage.

Example:
    from src.build_dp import main as build_dp_main
    build_dp_main()

    from src.generate_dc import main as generate_dc_main
    generate_dc_main()

    from src.data_filtering import process as filter_process
    filter_process("input.jsonl", "output.jsonl", "gemma3:1b")
"""

from .build_dp import main as build_dp_main
from .generate_dc import main as generate_dc_main
from .data_filtering import process as filter_process
from .data_filtering_2 import process as filter2_process
from .data_translate import process_file as translate_process
from .context import build_context
from .validate import parse_json_or_retry
from .llms import build_llm
from .translate_lib import get_translators
from .rubrics import EVAL_CODES, ACTION_CODES, STATE_CODES

__all__ = [
    "build_dp_main",
    "generate_dc_main",
    "filter_process",
    "filter2_process",
    "translate_process",
    "build_context",
    "parse_json_or_retry",
    "build_llm",
    "get_translators",
    "EVAL_CODES",
    "ACTION_CODES",
    "STATE_CODES",
]
