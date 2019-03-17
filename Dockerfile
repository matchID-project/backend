FROM python:2
ARG proxy
ENV http_proxy $proxy
ENV https_proxy $proxy

RUN pip install --upgrade pip 
RUN pip install `echo $proxy | sed 's/\(\S\S*\)/--proxy \1/'` simplejson tslib Flask Flask-OAuth flask-login enum34 flask_restplus rauth PyYAML nltk elasticsearch pandas Werkzeug scikit-learn[alldeps] geopy jellyfish networkx sqlalchemy psycopg2-binary redisearch

RUN mkdir -p /matchid/code /matchid/conf/run /matchid/log /matchid/referential_data /data/matchID_test/ /matchid/upload

WORKDIR /matchid

VOLUME /matchid/code
VOLUME /matchid/conf
VOLUME /matchid/projects
VOLUME /matchid/referential_data
VOLUME /matchid/log
VOLUME /matchid/models
VOLUME /matchid/upload

EXPOSE 8081

CMD ["./code/api.py"]



