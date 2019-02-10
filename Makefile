##############################################
# WARNING : THIS FILE SHOULDN'T BE TOUCHED   #
#    FOR ENVIRONNEMENT CONFIGURATION         #
# CONFIGURABLE VARIABLES SHOULD BE OVERRIDED #
# IN THE 'artifacts' FILE, AS NOT COMMITTED  #
##############################################

SHELL=/bin/bash
#matchID default exposition port
export PORT=8081

#matchID default paths
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
export DC_PREFIX=matchid
export DC_NETWORK=matchid
export DC_NETWORK_OPT=

export API_SECRET_KEY:=$(shell cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1 | sed 's/^/\*/;s/\(....\)/\1:/;s/$$/!/;s/\n//')
export ADMIN_PASSWORD:=$(shell cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1 | sed 's/^/\*/;s/\(....\)/\1:/;s/$$/!/;s/\n//' )
export ADMIN_PASSWORD_HASH:=$(shell echo -n ${ADMIN_PASSWORD} | sha384sum | sed 's/\s*\-.*//')
export POSTGRES_PASSWORD=matchid

export CRED_TEMPLATE=./creds.yml
export CRED_FILE=conf/security/creds.yml

# elasticsearch defaut configuration
export ES_NODES = 3		# elasticsearch number of nodes
export ES_SWARM_NODE_NUMBER = 2		# elasticsearch number of nodes
export ES_MEM = 1024m		# elasticsearch : memory of each node
export ES_VERSION = 6.5.3

dummy		    := $(shell touch artifacts)
include ./artifacts

commit              := $(shell git rev-parse HEAD | cut -c1-8)
lastcommit          := $(shell touch .lastcommit && cat .lastcommit)
commit-frontend     := $(shell (cd ${FRONTEND} 2> /dev/null) && git rev-parse HEAD | cut -c1-8)
lastcommit-frontend := $(shell (cat ${FRONTEND}/.lastcommit 2>&1) )
date                := $(shell date -I)
id                  := $(shell cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

vm_max_count		:= $(shell cat /etc/sysctl.conf | egrep vm.max_map_count\s*=\s*262144 && echo true)


PG := 'postgres'
DC := 'docker-compose'
include /etc/os-release



clean-secrets:
	rm ${CRED_FILE}

register-secrets: install-prerequisites
ifeq ("$(wildcard ${CRED_FILE})","")
	@echo WARNING new ADMIN_PASSWORD is ${ADMIN_PASSWORD}
	@envsubst < ${CRED_TEMPLATE} > ${CRED_FILE}
endif

install-prerequisites:
ifeq ("$(wildcard /usr/bin/envsubst)","")
	sudo apt-get update; true 
	sudo apt install -y gettext; true
endif

ifeq ("$(wildcard /usr/bin/docker)","")
	echo install docker-ce, still to be tested
	sudo apt-get update
	sudo apt-get install \
    	apt-transport-https \
	ca-certificates \
	curl \
	software-properties-common

	curl -fsSL https://download.docker.com/linux/${ID}/gpg | sudo apt-key add -
	sudo add-apt-repository \
		"deb https://download.docker.com/linux/ubuntu \
		`lsb_release -cs` \
   		stable"
	sudo apt-get update
	sudo apt-get install -y docker-ce
endif
	@(if (id -Gn ${USER} | grep -vc docker); then sudo usermod -aG docker ${USER}; fi) > /dev/null
ifeq ("$(wildcard /usr/local/bin/docker-compose)","")
	@echo installing docker-compose
	@sudo curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
	@sudo chmod +x /usr/local/bin/docker-compose
endif

docker-clean: stop
	docker container rm matchid-build-front matchid-nginx elasticsearch postgres kibana

clean:
	sudo rm -rf ${FRONTEND}/dist

network-stop:
	docker network rm ${DC_NETWORK}

network: install-prerequisites
	@docker network create ${DC_NETWORK_OPT} ${DC_NETWORK} 2> /dev/null; true

elasticsearch-stop:
	@echo docker-compose down matchID elasticsearch
ifeq "$(ES_NODES)" "1"
	@${DC} -f ${DC_FILE}-elasticsearch-phonetic.yml down
else
	@${DC} -f ${DC_FILE}-elasticsearch-huge.yml down
endif

elasticsearch2-stop:
	@${DC} -f ${DC_FILE}-elasticsearch-huge-remote.yml down



vm_max:
ifeq ("$(vm_max_count)", "")
	@echo updating vm.max_map_count $(vm_max_count) to 262144
	sudo sysctl -w vm.max_map_count=262144
endif

elasticsearch: network vm_max
	@echo docker-compose up matchID elasticsearch with ${ES_NODES} nodes
	@cat ${DC_FILE}-elasticsearch.yml | sed "s/%M/${ES_MEM}/g" > ${DC_FILE}-elasticsearch-huge.yml
	@(if [ ! -d ${BACKEND}/esdata/node1 ]; then sudo mkdir -p ${BACKEND}/esdata/node1 ; sudo chmod 777 ${BACKEND}/esdata/node1/.; fi)
	@(i=$(ES_NODES); while [ $${i} -gt 1 ]; \
		do \
			if [ ! -d ${BACKEND}/esdata/node$$i ]; then (echo ${BACKEND}/esdata/node$$i && sudo mkdir -p ${BACKEND}/esdata/node$$i && sudo chmod 777 ${BACKEND}/esdata/node$$i/.); fi; \
		cat ${DC_FILE}-elasticsearch-node.yml | sed "s/%N/$$i/g;s/%MM/${ES_MMEM}/g;s/%M/${ES_MEM}/g" >> ${DC_FILE}-elasticsearch-huge.yml; \
		i=`expr $$i - 1`; \
	done;\
	true)
	${DC} -f ${DC_FILE}-elasticsearch-huge.yml up -d

elasticsearch2:
	@echo docker-compose up matchID elasticsearch with ${ES_NODES} nodes
	@cat ${DC_FILE}-elasticsearch.yml | head -8 > ${DC_FILE}-elasticsearch-huge-remote.yml
	@(i=$$(( $(ES_NODES) * $(ES_SWARM_NODE_NUMBER) ));j=$$(( $(ES_NODES) * $(ES_SWARM_NODE_NUMBER) - $(ES_NODES))); while [ $${i} -gt $${j} ]; \
	        do \
	              if [ ! -d ${BACKEND}/esdata/node$$i ]; then (echo ${BACKEND}/esdata/node$$i && sudo mkdir -p ${BACKEND}/esdata/node$$i && sudo chmod 777 ${BACKEND}/esdata/node$$i/.); fi; \
	              cat ${DC_FILE}-elasticsearch-node.yml | sed "s/%N/$$i/g;s/%MM/${ES_MMEM}/g;s/%M/${ES_MEM}/g" | egrep -v 'depends_on|- elasticsearch' >> ${DC_FILE}-elasticsearch-huge-remote.yml; \
	              i=`expr $$i - 1`; \
	 	done;\
	true)
	${DC} -f ${DC_FILE}-elasticsearch-huge-remote.yml up -d

kibana-stop:
	${DC} -f ${DC_FILE}-kibana.yml down
kibana: network
ifeq ("$(wildcard ${BACKEND}/kibana)","")
	sudo mkdir -p ${BACKEND}/kibana && sudo chmod 777 ${BACKEND}/kibana/.
endif
	${DC} -f ${DC_FILE}-kibana.yml up -d

postgres-stop:
	${DC} -f ${DC_FILE}-${PG}.yml down
postgres: network
	${DC} -f ${DC_FILE}-${PG}.yml up -d
	docker exec ${DC_PREFIX}-postgres psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS fuzzystrmatch"

backend-stop:
	${DC} down
backend: network register-secrets
ifeq ("$(wildcard ${UPLOAD})","")
	@sudo mkdir -p ${UPLOAD}
endif
ifeq ("$(wildcard ${PROJECTS})","")
	@sudo mkdir -p ${PROJECTS}
endif
ifeq ("$(wildcard ${MODELS})","")
	@sudo mkdir -p ${PROJECTS}
endif
ifneq "$(commit)" "$(lastcommit)"
	@echo building matchID backend after new commit
	${DC} build
	@echo "${commit}" > ${BACKEND}/.lastcommit
endif
ifeq ("$(wildcard docker-compose-local.yml)","")
	${DC} up -d
else
	${DC} -f docker-compose.yml -f docker-compose-local.yml up -d
endif

frontend-download:
ifeq ("$(wildcard ${FRONTEND})","")
	@echo downloading frontend code
	@mkdir -p ${FRONTEND}
	@cd ${FRONTEND}; git clone https://github.com/matchID-project/frontend . #2> /dev/null; true
endif

frontend-update:
	@cd ${FRONTEND}; git pull

backend-update:
	@cd ${BACKEND}; git pull

update: frontend-update backend-update

frontend-dev:
ifneq "$(commit-frontend)" "$(lastcommit-frontend)"
	@echo docker-compose up matchID frontend for dev after new commit
	${DC} -f ${DC_FILE}-dev-frontend.yml up --build -d
	@echo "${commit-frontend}" > ${FRONTEND}/.lastcommit
else
	@echo docker-compose up matchID frontend for dev
	${DC} -f  ${DC_FILE}-dev-frontend.yml up -d
endif

frontend-dev-stop:
	${DC} -f ${DC_FILE}-dev-frontend.yml down

dev: network frontend-stop backend elasticsearch postgres frontend-dev

dev-stop: backend-stop kibana-stop elasticsearch-stop postgres-stop frontend-dev-stop newtork-stop

frontend-build: network frontend-download
ifneq "$(commit-frontend)" "$(lastcommit-frontend)"
	@echo building matchID frontend after new commit
	@make clean
	@sudo mkdir -p ${FRONTEND}/dist
	${DC} -f ${DC_FILE}-build-frontend.yml up --build
	@echo "${commit-frontend}" > ${FRONTEND}/.lastcommit
endif

frontend-stop:
	${DC} -f ${DC_FILE}-run-frontend.yml down

frontend: frontend-build
	@echo docker-compose up matchID frontend
	${DC} -f ${DC_FILE}-run-frontend.yml up -d

stop: backend-stop elasticsearch-stop kibana-stop postgres-stop
	@echo all components stopped

start-all: start postgres
	@sleep 2 && echo all components started, please enter following command to supervise:
	@echo tail log/docker-*.log

start: frontend-build elasticsearch backend frontend-stop frontend
	@sleep 2 && docker-compose logs

up: start

down: stop

restart: down up

log:
	@docker logs ${DC_PREFIX}-backend

example-download:
	@echo downloading example code
	@mkdir -p ${EXAMPLES}
	@cd ${EXAMPLES}; git clone https://github.com/matchID-project/examples . ; true
	@mv projects _${date}_${id}_projects 2> /dev/null; true
	@mv upload _${date}_${id}_upload 2> /dev/null; true
	@ln -s ${EXAMPLES}/projects ${BACKEND}/projects
	@ln -s ${EXAMPLES}/data ${BACKEND}/upload
