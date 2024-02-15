import argparse
import os
import subprocess
import sys
from datasets import load_dataset
from enum import Enum

INPUT_PATH = "data/imdb_train.csv"


class TestCase(Enum):
    EMBEDDINGS_BATCH1_CHUNK512 = {
        "name": "embeddings_batch1_chunk512",
        "batch_size": 1,
        "chunk_size": 512,
    }
    # EMBEDDINGS_BATCH1_CHUNK256 = {
    #     "name": "embeddings_batch1_chunk256",
    #     "batch_size": 1,
    #     "chunk_size": 256,
    # }
    # EMBEDDINGS_BATCH10_CHUNK512 = {
    #     "name": "embeddings_batch10_chunk512",
    #     "batch_size": 10,
    #     "chunk_size": 512,
    # }
    # EMBEDDINGS_BATCH10_CHUNK256 = {
    #     "name": "embeddings_batch10_chunk256",
    #     "batch_size": 10,
    #     "chunk_size": 256,
    # }
    # EMBEDDINGS_BATCH50_CHUNK512 = {
    #     "name": "embeddings_batch50_chunk512",
    #     "batch_size": 50,
    #     "chunk_size": 512,
    # }
    # EMBEDDINGS_BATCH50_CHUNK256 = {
    #     "name": "embeddings_batch50_chunk256",
    #     "batch_size": 50,
    #     "chunk_size": 256,
    # }
    EMBEDDINGS_BATCH100_CHUNK512 = {
        "name": "embeddings_batch100_chunk512",
        "batch_size": 100,
        "chunk_size": 512,
    }
    # EMBEDDINGS_BATCH100_CHUNK256 = {
    #     "name": "embeddings_batch100_chunk256",
    #     "batch_size": 100,
    #     "chunk_size": 256,
    # }


# Custom type function to convert input string to a list of integers
def int_list(value):
    try:
        return [int(item) for item in value.strip("[]").split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Value {value} is not a valid list of integers"
        )


def get_embedding_models():
    # return ["nemo_microservice"]
    return ["openai_ada002", "nemo_microservice"]


def run_suite(
    test_case: TestCase,
    loops=1,
    processes=1,
    report_dir=".",
    only_values_containing=None,
    threads_per_benchmark=None,
):
    if threads_per_benchmark is None:
        threads_per_benchmark: list[int] = [1]

    embedding_models = get_embedding_models()
    if only_values_containing is not None:
        for embedding_model in embedding_models:
            for filter_by in only_values_containing:
                if filter_by not in embedding_model:
                    embedding_models.remove(embedding_model)
                    break

    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(report_dir)

    filenames = []
    logs_file = os.path.join(args.reports_dir, "benchmarks.log")

    for embedding_model in embedding_models:
        for threads in threads_per_benchmark:
            test_name = test_case["name"]

            filename = f"{test_name}-{embedding_model}-{threads}.json"
            abs_filename = os.path.join(report_dir, filename)
            os.path.exists(abs_filename) and os.remove(abs_filename)
            filenames.append(abs_filename)

            batch_size = test_case["batch_size"]
            chunk_size = test_case["chunk_size"]
            command = f"{sys.executable} -m pyperf command --copy-env -p {processes} -n 1 -l {loops} -t -o {abs_filename} -- {sys.executable} {benchmarks_dir}/testcases.py {logs_file} {test_name} {embedding_model} {batch_size} {chunk_size} {threads}"
            print(
                f"Running suite: {test_name} with model: {embedding_model} and threads: {threads}"
            )
            try:
                subprocess.run(command.split(" "), text=True, check=True)
            except Exception as e:
                print(f"Error running suite: {e.args[0]}")
                if os.path.exists(logs_file):
                    with open(logs_file, "r") as f:
                        print(f.read())
                raise Exception("Error running suite")

    if len(filenames) <= 1:
        print("Not enough files to compare")
    else:
        filenames_str = " ".join(filenames)
        print("Showing comparison between files: {filenames_str}")

        comparison_command = (
            f"{sys.executable} -m pyperf compare_to --table -v {filenames_str}"
        )
        subprocess.run(comparison_command.split(" "), text=True, check=True)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmarks runner",
        description="Run benchmarks to compare different providers and combinations",
    )

    test_choices = ["all"]
    test_choices = test_choices + [t.value["name"] for t in TestCase]
    parser.add_argument(
        "-t",
        "--test-case",
        choices=test_choices,
        required=True,
        help="Test case to run",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="",
        help="Filter models to run (comma separated). e.g. to run only openai_ada002, use: openai_",
    )
    parser.add_argument(
        "-r",
        "--reports-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Reports dir",
    )

    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="The number of independent processes to run each benchmark. These run sequentially by default, and thus do not affect CPU/GPU access. Running multiple processes ensures sources of randomness (hash collisions, ASLR) do not affect results.",
    )

    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=1,
        help="Number of loops to run each benchmark. Results will be statistically computed over each loop.",
    )

    parser.add_argument(
        "-n",
        "--num_threads",
        type=int_list,
        default=[1],
        help="Number of threads (concurrent requests) per benchmark",
    )

    parser.add_argument(
        "--vector-database",
        type=str,
        default="none",
        help="If not 'none', the benchmark will store the generated embeddings in the given vector database",
    )

    args = parser.parse_args()
    if not os.path.exists(args.reports_dir):
        os.makedirs(args.reports_dir)
    print(f"Reports dir: {args.reports_dir}")

    if args.test_case == "all":
        tests_to_run = [t.value for t in TestCase]
    else:
        test_names_to_run = filter(None, args.test_case.split(","))
        tests_to_run = [
            test_case
            for name in test_names_to_run
            for test_case in TestCase
            if test_case.value["name"] == name
        ]

    logs_file = os.path.join(args.reports_dir, "benchmarks.log")
    if os.path.exists(logs_file):
        os.remove(logs_file)
    print(f"Logs file: {logs_file}")

    # Download the dataset to use
    if not os.path.exists(INPUT_PATH):
        directory = os.path.dirname(INPUT_PATH)
        if not os.path.exists(directory):
            os.makedirs(directory)

        dataset = load_dataset("imdb", split="train")
        dataset.to_csv(INPUT_PATH, index=False)
    print("Using dataset: ", INPUT_PATH)

    for test_case in tests_to_run:
        run_suite(
            test_case=test_case,
            report_dir=args.reports_dir,
            loops=args.loops,
            processes=args.processes,
            only_values_containing=args.models.split(","),
            threads_per_benchmark=args.num_threads,
        )
