import argparse

from main import main

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="Antivirus",
)

parser.add_argument(
    "--mode",
    default="automatic",
)

args = parser.parse_args()

main(
    dataset=args.dataset,
    mode=args.mode,
)