FROM python:2
ARG proxy
ENV http_proxy $proxy
ENV https_proxy $proxy

RUN pip install `echo $proxy | sed 's/\(\S\S*\)/--proxy \1/'` simplejson Flask Flask-OAuth flask-login flask_restplus rauth PyYAML nltk elasticsearch pandas Werkzeug scikit-learn[alldeps] geopy jellyfish feather-format tables networkx sqlalchemy psycopg2
 
RUN mkdir -p /matchid/code /matchid/conf/run /matchid/log /matchid/referential_data /data/matchID_test/

WORKDIR /matchid

VOLUME /matchid/code
VOLUME /matchid/conf
VOLUME /matchid/projects
VOLUME /matchid/referential_data
VOLUME /matchid/log
VOLUME /matchid/models

EXPOSE 8081

CMD ["./code/api.py"]



