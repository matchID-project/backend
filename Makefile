export BACKEND := $(shell pwd)

export FRONTEND=${BACKEND}/../frontend
export UPLOAD=${BACKEND}/upload
export PROJECTS=${BACKEND}/projects
export EXAMPLES=${BACKEND}/../examples
export TUTORIAL=${BACKEND}/../tutorial
export MODELS=${BACKEND}/models
export LOG=${BACKEND}/log
export DC_DIR=${BACKEND}/docker-components
export DC_FILE=${DC_DIR}/docker-compose

commit              := $(shell git rev-parse HEAD | cut -c1-8)
lastcommit          := $(shell touch .lastcommit; cat .lastcommit)
commit-frontend     := $(shell cd ${FRONTEND}; git rev-parse HEAD | cut -c1-8)
lastcommit-frontend := $(shell touch ${FRONTEND}/.lastcommit; cat ${FRONTEND}/.lastcommit)
date                := $(shell date -I)
id                  := $(shell cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

ES_NODES := 1
PG := 'postgres'
DC := 'docker-compose'
NH := 'nohup'


docker-clean: stop
	docker container rm matchid-build-front matchid-nginx elasticsearch postgres kibana 

clean:
	sudo rm -rf ${FRONTEND}/dist
	sudo mkdir -p ${UPLOAD} ${PROJECTS} ${MODELS}

network-stop:
	docker network rm matchid

network:
	@docker network create matchid 2> /dev/null; true

elasticsearch-stop:
	@echo docker-compose down matchID elasticsearch
	@${DC} -f ${DC_FILE}-elasticsearch-phonetic.yml down 
	@${DC} -f ${DC_FILE}-elasticsearch-huge.yml down

elasticsearch: network
ifeq "$(ES_NODES)" "1"
	@echo docker-compose up matchID elasticsearch
	@sudo mkdir -p esdata/node
	@(${NH} ${DC} -f ${DC_FILE}-elasticsearch-phonetic.yml up > ${LOG}/docker-elasticsearch.log) 2> /dev/null & 
else
	@echo docker-compose up matchID elasticsearch with ${ES_NODES} nodes
	@cat ${DC_FILE}-elasticsearch.yml > ${DC_FILE}-elasticsearch-huge.yml
	@i=2; $while [$i -le $ES_NODES]; do cat ${DC_FILE}-elasticsearch-node.yml | sed "s/%N/$i" >> ${DC_FILE}-elasticsearch-huge.yml;done
	@(${NH} ${DC} -f ${DC_FILE}-elasticsearch-huge.yml up > ${LOG}/docker-elasticsearch.log) 2> /dev/null & 
endif
	@sleep 2 && tail ${LOG}/docker-elasticsearch.log &

kibana-stop:
	${DC} -f ${DC_FILE}-kibana.yml down
kibana: network
	@echo docker-compose up matchID kibana
	@(${NH} ${DC} -f ${DC_FILE}-kibana.yml up > ${LOG}/docker-kibana.log) 2> /dev/null & 
	@sleep 2 && tail ${LOG}/docker-kibana.log &

postgres-stop:
	${DC} -f ${DC_FILE}-${PG}.yml down
postgres: network
	@echo docker-compose up matchID postgres
	@(${NH} ${DC} -f ${DC_FILE}-${PG}.yml up > ${LOG}/docker-postgres.log) 2> /dev/null & 
	@sleep 2 && tail ${LOG}/docker-postgres.log &

backend-stop:
	${DC} down 
backend: network
	@echo docker-compose up matchID backend
	@(${NH} ${DC} up > ${LOG}/docker-backend.log) 2> /dev/null & 
	@sleep 2 && tail ${LOG}/docker-backend.log &

frontend-download:
	@echo downloading frontend code
	@mkdir -p ${FRONTEND}
	@cd ${FRONTEND}; git clone https://github.com/matchID-project/frontend . 2> /dev/null; true 
	@cd ${BACKEND}

start-dev: frontend-download network backend elasticsearch postgres kibana
ifneq "$(commit-frontend)" "$(lastcommit-frontend)"
	@echo docker-compose up matchID frontend for dev after new commit
	@${NH} ${DC} -f docker/docker-compose-dev.yml down
	@(${NH} ${DC} -f docker/docker-compose-dev.yml up --build > ${LOG}/docker-frontend.log) 2> /dev/null & 
	@echo "${commit-frontend}" > ${FRONTEND}/.lastcommit
else
	@echo docker-compose up matchID frontend for dev
	@(${NH} ${DC} -f docker-compose-dev.yml up > ${LOG}/docker-frontend.log) 2> /dev/null & 
endif
	@sleep 2 && tail ${LOG}/docker-frontend.log &

frontend-build: frontend-download network
ifneq "$(commit-frontend)" "$(lastcommit-frontend)"
	@echo building matchID frontend after new commit
	@make clean
	@echo building frontend in ${FRONTEND}
	@sudo mkdir -p ${FRONTEND}/dist
	${DC} -f ${DC_FILE}-build-frontend.yml up --build
	@echo "${commit-frontend}" > ${FRONTEND}/.lastcommit
endif	

frontend-stop:
	${DC} -f ${DC_FILE}-run-frontend.yml down

frontend: frontend-build
	@echo docker-compose up matchID frontend
	@(${NH} ${DC} -f ${DC_FILE}-run-frontend.yml up > ${LOG}/docker-frontend.log) 2> /dev/null & 
	@sleep 2 && tail ${LOG}/docker-frontend.log &

stop: backend-stop elasticsearch-stop kibana-stop postgres-stop 
	@echo all components stopped

start-all: start postgres
	@sleep 2 && echo all components started, please enter following command to supervise: 
	@echo tail log/docker-*.log

start: backend elasticsearch kibana frontend
	@sleep 2 && echo essential components started, please enter following command to supervise: 
	@echo tail log/docker-*.log

example-download:
	@echo downloading example code
	@mkdir -p ${EXAMPLES}
	@cd ${EXAMPLES}; git clone https://github.com/matchID-project/examples . ; true
	@mv projects _${date}_${id}_projects 2> /dev/null; true
	@mv upload _${date}_${id}_upload 2> /dev/null; true
	@ln -s ${EXAMPLES}/projects ${BACKEND}/projects
	@ln -s ${EXAMPLES}/data ${BACKEND}/upload

tuto: start
	@mkdir -p ${TUTORIAL}/projects ${TUTORIAL}/data ${TUTORIAL}/models 
	@mv projects _${date}_${id}_projects 2> /dev/null; true
	@mv upload _${date}_${id}_upload 2> /dev/null; true
	@mv models _${date}_${id}_models 2> /dev/null; true
	@ln -s ${TUTORIAL}/projects ${BACKEND}/projects
	@ln -s ${TUTORIAL}/data ${BACKEND}/upload
	@ln -s ${TUTORIAL}/models ${BACKEND}/models

