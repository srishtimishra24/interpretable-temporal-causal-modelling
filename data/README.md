# Dataset Organisation

The datasets used in this project are not included in this repository.

The IT monitoring datasets were obtained from the public dataset repository associated with:

> Aït-Bachir, A., Assaad, C. K., de Bignicourt, C., Devijver, E., Ferreira, S., Gaussier, E., Mohanna, H., and Zan, L. (2023). *Case Studies of Causal Discovery from IT Monitoring Time Series*.

The original IT Monitoring benchmark repository is:

https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series

The accompanying paper is available at:

https://arxiv.org/abs/2307.15678

## Directory Structure

After obtaining the datasets, organise the files using the following structure:

```text
data/
├── Antivirus_Activity/
│   ├── combined.csv
│   └── structure.txt
│
├── Web_Activity/
│   ├── combined.csv
│   └── structure.txt
│
├── Middleware_oriented_message_Activity/
│   ├── combined.csv
│   └── structure.txt
│
└── Storm_Ingestion_Activity/
    ├── storm_data_normal.csv
    └── storm_structure.txt
```

## Dataset Files

Each dataset directory contains:

- `combined.csv` or `storm_data_normal.csv` -- the time-series data used by the causal discovery pipeline.
- `structure.txt` or `storm_structure.txt` -- the corresponding ground-truth causal structure used for evaluation.

The `combined.csv` files used in this project were created by combining the corresponding preprocessed data files provided for the respective datasets.

## Datasets

The framework was evaluated on four IT monitoring datasets:

- Antivirus Activity
- Web Activity
- Middleware Oriented Message Activity
- Storm Ingestion Activity

The datasets are publicly available through the original IT Monitoring benchmark repository and are not redistributed with this project.

## Running the Pipeline

Once the datasets have been organised, the command-line interface can be used to run causal discovery.

For example:

```bash
python src/cli.py \
--input data/Antivirus_Activity/combined.csv \
--output outputs/Antivirus_final_TM.txt \
--ground-truth data/Antivirus_Activity/structure.txt \
--tau 3 \
--runs 10 \
--stability 0.7 \
--top-k 3 \
--clauses 500 \
--epochs 100
```

The same structure can be used for the other datasets by changing the input, output, and ground-truth paths.

## Reference

```bibtex
@article{kassaad2023itmonitoring,
  title={Case Studies of Causal Discovery from IT Monitoring Time Series},
  author={A{\"i}t-Bachir, Ali and Assaad, Charles K. and de Bignicourt, Christophe and Devijver, Emilie and Ferreira, Simon and Gaussier, Eric and Mohanna, Hosein and Zan, Lei},
  year={2023},
  eprint={2307.15678},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```