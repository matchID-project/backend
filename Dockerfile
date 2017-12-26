FROM python:2
ARG proxy

RUN pip install `echo $proxy | sed 's/\(\S\S*\)/--proxy \1/'` simplejson Flask flask_restplus PyYAML nltk elasticsearch pandas Werkzeug scikit-learn[alldeps] geopy jellyfish feather-format tables python-igraph
 
RUN mkdir -p /matchid/code /matchid/conf/run /matchid/log /matchid/referential_data /data/matchID_test/

WORKDIR /matchid

VOLUME /matchid/code
VOLUME /matchid/conf
VOLUME /matchid/projects
VOLUME /matchid/referential_data
VOLUME /matchid/log
VOLUME /matchid/models

EXPOSE 8081

CMD ["./code/recipe.py"]



