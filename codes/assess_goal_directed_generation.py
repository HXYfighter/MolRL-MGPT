import datetime
import json
from collections import OrderedDict
from typing import List, Any, Dict

import guacamol
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark, GoalDirectedBenchmarkResult
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.data import get_time_string


def assess_goal_directed_generation(goal_directed_molecule_generator: GoalDirectedGenerator,
                                    json_output_file='output_goal_directed.json',
                                    benchmark_version='v2', task_id=0) -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    if task_id in list(range(20)):
        benchmarks = [goal_directed_benchmark_suite(version_name=benchmark_version)[task_id]]
    elif task_id == -1:
        benchmarks = goal_directed_benchmark_suite(version_name=benchmark_version)

    results = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmarks=benchmarks)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_goal_directed_benchmarks(goal_directed_molecule_generator: GoalDirectedGenerator,
                                       benchmarks: List[GoalDirectedBenchmark]
                                       ) -> List[GoalDirectedBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        goal_directed_molecule_generator: model to assess
        benchmarks: list of benchmarks to evaluate
    """

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        print(f'Running benchmark: {benchmark.name}')
        result = benchmark.assess_model(goal_directed_molecule_generator)
        print(f'Results for the benchmark {result.benchmark_name}:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}')
        print(f'  Metadata: {result.metadata}')
        results.append(result)

    return results
