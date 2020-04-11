import werkzeug
from flask_restplus import reqparse
from werkzeug.datastructures import FileStorage


conf_parser = reqparse.RequestParser()
conf_parser.add_argument('file', location='files',
                         type=FileStorage, action='append', required=True, help='yaml file')

yaml_parser = reqparse.RequestParser()
yaml_parser.add_argument('yaml',
                         type=str, required=True, help='yaml text')

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, action='append', required=True, help='data file : csv, ...')
live_parser = reqparse.RequestParser()
live_parser.add_argument('file', location='files',
                         type=FileStorage, required=True, help='data file : csv, ...')

es_parser = reqparse.RequestParser()
es_parser.add_argument('size', location='args',
                       type=int, required=False, help='limit size of search')


login_parser = reqparse.RequestParser()
login_parser.add_argument('json', location='json',
                          type='json', required=True, help='json')

download_parser = reqparse.RequestParser()
download_parser.add_argument('size', location='args',
                       type=int, required=False, help='limit size of search')
download_parser.add_argument('type', location='args',
                       type=str, required=False, help='type: csv, default json')
