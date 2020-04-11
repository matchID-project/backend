#######################
# Step 1: Base target #
#######################
FROM python:3-slim as base
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG APP

RUN apt-get update -y
RUN apt-get install curl -y

RUN pip install --upgrade pip

WORKDIR /${APP}

RUN ls

COPY requirements.txt .
RUN pip install `echo $http_proxy | sed 's/\(\S\S*\)/--proxy \1/'` -r requirements.txt

RUN mkdir -p code\
             conf/run\
             log\
             referential_data\
             matchID_test/\
             upload

################################
# Step 2: "production" target #
################################
FROM base as production
ARG APP
ENV BACKEND_PORT ${BACKEND_PORT}
ENV BACKEND_DEBUG False
ENV BACKEND_RELOAD False
ENV APP ${APP}

WORKDIR /${APP}

COPY code/ code/
COPY conf/ conf/
COPY referential_data/ referential_data/

VOLUME /${app_path}/projects
VOLUME /${app_path}/log
VOLUME /${app_path}/models
VOLUME /${app_path}/upload

EXPOSE ${BACKEND_PORT}

CMD ["./code/api.py"]

################################
# Step 3: "development" target #
################################
FROM production as development
ARG APP
ENV BACKEND_PORT ${BACKEND_PORT}
ENV APP ${APP}
ENV BACKEND_DEBUG True
ENV BACKEND_RELOAD True

VOLUME /${APP}/code
VOLUME /${APP}/conf
VOLUME /${APP}/projects
VOLUME /${APP}/referential_data
VOLUME /${APP}/log
VOLUME /${APP}/models
VOLUME /${APP}/upload

EXPOSE ${BACKEND_PORT}

CMD ["./code/api.py"]



