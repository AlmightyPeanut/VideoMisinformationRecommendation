import importlib

from elliot.run import run_experiment


def main():
    run_experiment("elliot_configs/non_hybrid_experiments.yml")


if __name__ == '__main__':
    main()
