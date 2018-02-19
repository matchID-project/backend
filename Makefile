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
lastcommit          := $(shell touch .lastcommit && cat .lastcommit)
commit-frontend     := $(shell (cd ${FRONTEND} 2> /dev/null) && git rev-parse HEAD | cut -c1-8)
lastcommit-frontend := $(shell (cat ${FRONTEND}/.lastcommit 2>&1) )
date                := $(shell date -I)
id                  := $(shell cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

vm_max_count		:= $(shell cat /etc/sysctl.conf | egrep vm.max_map_count\s*=\s*262144 && echo true)

# Elasticsearch configuration
# Nuber of nodes, memory, and container memory (used only for many nodes)
ES_NODES := 3
ES_MEM := 1024m
ES_MMEM := 2048m

PG := 'postgres'
DC := 'docker-compose'
include /etc/os-release 

install-prerequisites:
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
      if (id -Gn ${USER} | grep -vc docker); then sudo usermod -aG docker ${USER} ;fi
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
	docker network rm matchid

network: install-prerequisites
	@docker network create matchid 2> /dev/null; true

elasticsearch-stop:
	@echo docker-compose down matchID elasticsearch
	@${DC} -f ${DC_FILE}-elasticsearch-phonetic.yml down 
	@${DC} -f ${DC_FILE}-elasticsearch-huge.yml down

vm_max:
ifeq ("$(vm_max_count)", "")
	@echo updating vm.max_map_count $(vm_max_count) to 262144
	sudo sysctl -w vm.max_map_count=262144
endif

elasticsearch: network vm_max
ifeq ("$(wildcard ${BACKEND}/esdata/node)","")
	sudo mkdir -p ${BACKEND}/esdata/node && sudo chmod 777 ${BACKEND}/esdata/node/.
endif
ifeq "$(ES_NODES)" "1"
	${DC} -f ${DC_FILE}-elasticsearch-phonetic.yml up --build -d
else
	@echo docker-compose up matchID elasticsearch with ${ES_NODES} nodes
	@cat ${DC_FILE}-elasticsearch.yml | sed "s/%MM/${ES_MMEM}/g;s/%M/${ES_MEM}/g" > ${DC_FILE}-elasticsearch-huge.yml
	@sudo mkdir -p ${BACKEND}/esdata/node1 && sudo chmod 777 ${BACKEND}/esdata/node1/.
	@i=$(ES_NODES); while [ $${i} -gt 1 ]; do \
		sudo mkdir -p ${BACKEND}/esdata/node$$i && sudo chmod 777 ${BACKEND}/esdata/node$$i/. ; \
		cat ${DC_FILE}-elasticsearch-node.yml | sed "s/%N/$$i/g;s/%MM/${ES_MMEM}/g;s/%M/${ES_MEM}/g" >> ${DC_FILE}-elasticsearch-huge.yml; \
		i=`expr $$i - 1`; \
	done;\
	true
	${DC} -f ${DC_FILE}-elasticsearch-huge.yml up -d 
endif

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

backend-stop:
	${DC} down 
backend: network
ifeq ("$(wildcard ${UPLOAD})","")
	@sudo mkdir -p ${UPLOAD}
endif
ifeq ("$(wildcard ${PROJECTS})","")
	@sudo mkdir -p ${PROJECTS}
endif
ifeq ("$(wildcard ${MODELS})","")
	@sudo mkdir -p ${PROJECTS}
endif
	${DC} up -d 

frontend-download:
ifeq ("$(wildcard ${FRONTEND})","")
	@echo downloading frontend code
	@mkdir -p ${FRONTEND}
	@cd ${FRONTEND}; git clone https://github.com/matchID-project/frontend . #2> /dev/null; true 
endif

start-dev: network backend elasticsearch postgres kibana
ifneq "$(commit-frontend)" "$(lastcommit-frontend)"
	@echo docker-compose up matchID frontend for dev after new commit
	${DC} -f ${DC_FILE}-dev-frontend.yml up --build -d 
	@echo "${commit-frontend}" > ${FRONTEND}/.lastcommit
else
	@echo docker-compose up matchID frontend for dev
	${DC} -f  ${DC_FILE}-dev-frontend.yml up -d 
endif

frontend-build: frontend-download network
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

start: frontend-download elasticsearch kibana backend frontend
	@sleep 2 && docker-compose logs

log:
	@docker logs matchid-backend

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

