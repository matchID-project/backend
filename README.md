# Introduction

This project aims to offer a backend to the matchID project.

The main objective is to process one or many datasets of civil states and identify multiple matches (at least two!) of a same person.

The backend basically offers the possibility to cook a dataset with a recipe, leading to a new dataset.
The recipe can be cooked "live" or in background, offering the possibility of **live-reranking** (with **machine learning** or not) on top of an elasticsearch.

A recipe book for preparing names, birth location, fuzzy match an rescore is integrated and can be fully customized for your use-case.

It's **full-api** designed (no cli!) and based on **Flask RESTPlus**.
The scalability relies on single server multiprocessing for the **Pandas** adn **scikit-learn** python part,
and cloud scalability of **elasticsearch** for large fuzzy-match use-cases.
It aims to offer capability to match two datasets with dozens millions of records in a day on a 1U server. Further developments will be still needed for full-cloud scalability.

For now the code is considered to be still in "alpha" development, and still needs some steps of refactoring and documentation to reach production readiness.

This package integrates a simple js web-app for helping developing your use-case (single user designed, so not to deserve a Lab of data-scientists),
and a **Docker** configuration for accelerating your use-case design.


# Main use cases

- live search and bulk-search identities in a dataset (take benefit from elasticsearch and offers the possibility of re-ranking)
- find common identities between two datasets
- deduplicate idendities into one dataset

# Running it
Automatization (and thus documentation) is for now a future achievement

First clone the project
```
git clone https://github.com/eig-2017/matchID-backend.git
```

You have to clone matchID-frontend project to enable annotation for machine learning capabilities:
```
git clone https://github.com/eig-2017/matchID-frontend.git
```

And the matchid-examples to have some data and recipes to play with:
```
git clone https://github.com/eig-2017/matchID-examples.git
```

Note that machine learning is not mandatory (you can have a real serious matching only based on rules) but seriously recommended for reduction development time.

Simply run it with Docker (a >8Go configuration is recommended)
```
cd matchID-backend
export FRONTEND=../matchID-frontend    # path to GitHub clone of matchID-frontend
export PROJECTS=../matchID-examples/projects/       # path to projects
export UPLOAD=../matchID-examples/data/       # path to upload
export MODELS=../matchID-examples/models       # path to upload

docker-compose -f docker-compose-dev.yml up --build
```

Which launches four containers :
- nginx (for static web files to test the backend)
- matchid-frontend (vuejs frontend)
- matchid-backend (python backend)
- elasticsearch (the database)

This configuration is intented for quick discovery and configuration and should not be ported as such into production.


We recommand to follow then the [**tutorial**](docs/tutorial.md) for a first usecase.
