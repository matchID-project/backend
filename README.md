# Introduction

This project aims to offer a backend to the matchID project.

Full documentation is available at [https://matchid-project.github.io](https://matchid-project.github.io/).

The main objective is to process one or many datasets of civil states and identify multiple matches (at least two!) of a same person.

The backend basically offers the possibility to cook a dataset with a recipe, leading to a new dataset.
The recipe can be cooked "live" or in background, offering the possibility of **live-reranking** (with **machine learning** or not) on top of an elasticsearch.

A recipe book for preparing names, birth location, fuzzy match an rescore is integrated and can be fully customized for your use-case.

It's **full-api** designed (no cli!) and based on **Flask RESTPlus**.
The scalability relies on single server multiprocessing for the **Pandas** adn **scikit-learn** python part,
and cloud scalability of **elasticsearch** for large fuzzy-match use-cases.
It aims to offer capability to match two datasets with dozens millions of records in a day on a 1U server. Further developments will be still needed for full-cloud scalability.

For now the code is considered to be still in "beta" development, and still needs some steps of refactoring and documentation to reach production readiness.

This package integrates a simple page application in Vue web-app for helping developing your use-case (single user designed, so not to deserve a Lab of data-scientists), and a **Docker** configuration for accelerating your use-case design.


# Main use cases

- live search and bulk-search identities in a dataset (take benefit from elasticsearch and offers the possibility of re-ranking)
- find common identities between two datasets
- deduplicate idendities into one dataset

# Running it
Automatization (and thus documentation) is for now a future achievement

First clone the backend
```
git clone https://github.com/matchID-project/backend
```

matchID uses `make` and Docker to accelerate installation of dependencies. You'll first have to [install Docker](https://docs.docker.com/engine/installation/) and [`docker-compose`](https://docs.docker.com/compose/).

Now you just have to start matchID:

``̀`
cd backend
make start
```

This should :
- download the frontend
- build it
- start the backend
- start elasticsearch (required)
- start kibana (optional, useful for exploring data insterted in elasticsearch)
- start postgres (optional)

If you have not enough memory (less than 12Go), we recommand not to start postgres neirther kibana or stop them :

```
make postgres stop
make kibana stop
```

# Follow the tutorial !

You can now add you own data, but we strongly to follow the tutorial and downloading the sample use case :

```
make download-example
```


We recommand now you to follow the [Tutorial](https://matchid-project.github.io/tutorial).


