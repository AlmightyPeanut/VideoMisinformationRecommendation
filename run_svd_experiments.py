import importlib

from elliot.run import run_experiment


def main():
    run_experiment("elliot_configs/svd_experiments.yaml")


if __name__ == '__main__':
    main()
