# evaluation_lib/ragas_evaluator.py

import logging
from typing import List, cast

import numpy as np
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from .models import ExperimentResult, QueryEvaluationResult
from .utils import categorize_query

logger = logging.getLogger(__name__)


def evaluate_single_query(
    experiment_result: ExperimentResult,
    contexts: List[str],
    llm_model: str,
    embedding_model: str,
    dry_run: bool = False,
) -> QueryEvaluationResult:
    """
    Evaluate a single ExperimentResult using RAGAS.

    Args:
        experiment_result: Minimal experiment result
        contexts: Retrieved chunk texts (empty for E1)
        llm_model: Model name for RAGAS LLM (e.g., "gpt-4o-mini")
        embedding_model: Model name for RAGAS embeddings (e.g., "text-embedding-3-small")
        dry_run: Skip API calls and return mock scores

    Returns:
        QueryEvaluationResult with CP, F, AR scores and computed metrics
    """
    if dry_run:
        # Generate mock scores for testing
        cp = np.random.uniform(0.5, 0.9)
        f = np.random.uniform(0.5, 0.9)
        ar = np.random.uniform(0.5, 0.9)
        logger.debug(
            f"DRY_RUN: Mock scores for {experiment_result.query_id}: CP={cp:.3f}, F={f:.3f}, AR={ar:.3f}"
        )
    else:
        # Real RAGAS evaluation
        scores = _evaluate_with_ragas(
            question=experiment_result.query,
            answer=experiment_result.llm_answer,
            contexts=contexts,
            ground_truth=experiment_result.ground_truth,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
        cp, f, ar = scores

    # Compute derived metrics
    geometric_mean = (cp * f * ar) ** (1 / 3)
    hallucination_risk_index = (1 - f) * ar

    # Categorize
    category = categorize_query(cp, f, ar)

    # Create result
    evaluation_result = QueryEvaluationResult(
        query_id=experiment_result.query_id,
        experiment=experiment_result.experiment,
        context_precision=cp,
        faithfulness=f,
        answer_relevancy=ar,
        geometric_mean=geometric_mean,
        hallucination_risk_index=hallucination_risk_index,
        category=category,
    )

    logger.debug(
        f"Evaluated {experiment_result.query_id}: {category} (GMean={geometric_mean:.3f})"
    )

    return evaluation_result


def evaluate_batch(
    experiment_results: List[ExperimentResult],
    contexts_list: List[List[str]],
    llm_model: str,
    embedding_model: str,
    batch_size: int = 10,
    dry_run: bool = False,
    rate_limit_delay: float = 1.0,
) -> List[QueryEvaluationResult]:
    """
    Evaluate multiple ExperimentResults using RAGAS in batches.

    Args:
        experiment_results: List of minimal experiment results
        contexts_list: List of context lists (one per query)
        llm_model: Model name for RAGAS LLM
        embedding_model: Model name for RAGAS embeddings
        batch_size: Number of queries to evaluate per batch
        dry_run: Skip API calls
        rate_limit_delay: Delay between batches to avoid rate limits

    Returns:
        List of QueryEvaluationResult objects
    """
    if len(experiment_results) != len(contexts_list):
        raise ValueError("experiment_results and contexts_list must have same length")

    if not experiment_results:
        return []

    all_results = []

    # Process in batches
    for i in range(0, len(experiment_results), batch_size):
        batch_end = min(i + batch_size, len(experiment_results))
        batch_results = experiment_results[i:batch_end]
        batch_contexts = contexts_list[i:batch_end]

        logger.info(
            f"Evaluating batch {i // batch_size + 1}: queries {i + 1}-{batch_end}"
        )

        if dry_run:
            # Mock evaluation for entire batch
            batch_evaluation_results = []
            for result, contexts in zip(batch_results, batch_contexts):
                eval_result = evaluate_single_query(
                    result, contexts, llm_model, embedding_model, dry_run=True
                )
                batch_evaluation_results.append(eval_result)
        else:
            # Real batch evaluation using RAGAS
            batch_evaluation_results = _evaluate_batch_with_ragas(
                batch_results, batch_contexts, llm_model, embedding_model
            )

        all_results.extend(batch_evaluation_results)

        # Rate limiting between batches
        if i + batch_size < len(experiment_results) and not dry_run:
            logger.debug(f"Rate limiting: sleeping {rate_limit_delay}s")
            import time

            time.sleep(rate_limit_delay)

    logger.info(f"Completed evaluation of {len(all_results)} queries")
    return all_results


def _evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    llm_model: str,
    embedding_model: str,
) -> tuple[float, float, float]:
    """
    Internal function to evaluate a single query using RAGAS.

    Returns: (context_precision, faithfulness, answer_relevancy)
    """
    try:
        # Create dataset in RAGAS format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # Configure LLM and embeddings
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

        # Configure metrics
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

        metrics = [ContextPrecision(), Faithfulness(), AnswerRelevancy()]

        # Evaluate
        raw_results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,  # Return NaN on failure
        )

        # Cast to EvaluationResult for proper type hints
        results = cast(EvaluationResult, raw_results)

        # Extract scores using to_pandas() method
        df = results.to_pandas()
        cp = float(df["context_precision"].iloc[0])
        f = float(df["faithfulness"].iloc[0])
        ar = float(df["answer_relevancy"].iloc[0])

        # Handle NaN values by replacing with 0.0 (worst score)
        if np.isnan(cp):
            cp = 0.0
        if np.isnan(f):
            f = 0.0
        if np.isnan(ar):
            ar = 0.0

        return cp, f, ar

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        # Return worst scores on failure
        return 0.0, 0.0, 0.0


def _evaluate_batch_with_ragas(
    batch_results: List[ExperimentResult],
    batch_contexts: List[List[str]],
    llm_model: str,
    embedding_model: str,
) -> List[QueryEvaluationResult]:
    """
    Internal function to evaluate a batch of queries using RAGAS.

    Returns list of QueryEvaluationResult objects.
    """
    try:
        # Create dataset in RAGAS format for batch
        data = {
            "question": [r.query for r in batch_results],
            "answer": [r.llm_answer for r in batch_results],
            "contexts": batch_contexts,
            "ground_truth": [r.ground_truth for r in batch_results],
        }
        dataset = Dataset.from_dict(data)

        # Configure LLM and embeddings
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

        # Configure metrics
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

        metrics = [ContextPrecision(), Faithfulness(), AnswerRelevancy()]

        # Evaluate batch
        raw_results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,  # Return NaN on failure
        )

        # Cast to EvaluationResult for proper type hints
        results = cast(EvaluationResult, raw_results)

        # Extract scores using to_pandas() method
        df = results.to_pandas()

        # Process results
        batch_evaluation_results = []
        for i, result in enumerate(batch_results):
            # Extract scores (handle potential NaN values)
            cp = float(df["context_precision"].iloc[i])
            f = float(df["faithfulness"].iloc[i])
            ar = float(df["answer_relevancy"].iloc[i])

            # Handle NaN values by replacing with 0.0 (worst score)
            if np.isnan(cp):
                cp = 0.0
            if np.isnan(f):
                f = 0.0
            if np.isnan(ar):
                ar = 0.0

            # Compute derived metrics
            geometric_mean = (cp * f * ar) ** (1 / 3)
            hallucination_risk_index = (1 - f) * ar
            category = categorize_query(cp, f, ar)

            evaluation_result = QueryEvaluationResult(
                query_id=result.query_id,
                experiment=result.experiment,
                context_precision=cp,
                faithfulness=f,
                answer_relevancy=ar,
                geometric_mean=geometric_mean,
                hallucination_risk_index=hallucination_risk_index,
                category=category,
            )

            batch_evaluation_results.append(evaluation_result)

        return batch_evaluation_results

    except Exception as e:
        logger.error(f"Batch RAGAS evaluation failed: {e}")
        raise
