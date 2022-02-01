### This repo describes experimental details on Wikitext-103 benchmark
#### 1. Data Preparation:
To download the data, please follow the instructions [here](https://github.com/yxuansu/SimCTG/tree/main/data).

The dataset contains three files:
    ├── wikitext103                    
    │   ├── wikitext103_raw_v1_train.txt          # Training Set
    │   ├── wikitext103_raw_v1_validation.txt     # Validation Set
    │   └── wikitext103_raw_v1_test.txt           # Test Set
    └── ...

#### 2. Train SimCTG:

#### 3. Decode Result:

> **A:** Because you don't want to test the code, you want to test the *program*.

    .
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...
