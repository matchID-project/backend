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

```
cd backend
make start
```

This should :
- download the frontend
- build it
- start the backend
- start elasticsearch (required)

Going to your [browser](http://localhost) to check everythings works fine.

As it starts many components your computer may encounter low memory [if <8Go]. Just go to next section to see how to still run matchID.

You can now add you own data, but we strongly to follow the tutorial and downloading the sample use case :

```
make download-example
```

We recommand now you to follow the [Tutorial](https://matchid-project.github.io/tutorial).

# Frequent running problems

### stop matchID

```
make stop
```

### supported components
The list of the supported components is :
- `backend` : the api and the engine
- `frontend`: the single-page-application in Vue.js to develop recipes
- `elasticsearch`: the famous search engine used for fuzzy matching
- `postgres` : not needed but useful for lower memory configuration in further 
- `kibana`: useful for elasticsearch data analysis

You can start all the components like this: 
```
make start-all
```

Each component can be started or stopped alone, this will for example stop postgres:
```
make postgres-stop
```

And this will start kibana :
```
make kibana
```

For example, `make start` is equivalent to `make backend frontend elasticsearch`.


### check health of components
Each docker components logs its actions to `log/docker-component.log`. So you can easily check heath of all components like this:

```
tail -f log/docker-*.log
```


### Nginx didn't launch
matchID use the 80 port by default. If you have another web service it may cause conflict. 

Just edit `docker-components/docker-compose-run-frontend.yml` and change the docker ports "80:80" to "8080:80" to change the exposition port to 8080.

You will restart the frontend like this

```
make frontend-stop frontend
```

### Clean all and retry

To clean everything 
```
make clean docker-clean
```

### Developpement mode
If you want to contribute to the developpement, you'll be able to fork the repo on GitHub and to lauch the dev mode (you'll perhaps have to do a `make docker-clean` first): 

```
make start-dev
```

### Running tests
Install dependencies and run tests with Make:
```bash
make tests
```
You can also install the requirements and run pytest directly:
```bash
pip install -r requirements.txt
pytest
```
