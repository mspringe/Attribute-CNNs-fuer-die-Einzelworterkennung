"""
This module provides a script to extract data from all JSON files stored in a specific directory and create a HTML
table for an better overview of the data.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from collections import defaultdict
from argparse import ArgumentParser

import os
import sys
import json
import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
sys.path.append(FILE_DIR)
from src.util import sanity_util


def jsons_to_table(dir_jsons, dir_out, name, format='html'):
    """
    Extracts the informations stored in the JSON files and stores creates an  HTML-table for them.

    :param dir_jsons: directory of JSON files
    :param dir_out: output directory of the HTML-table
    :param name: name of the HTML page
    """
    # sanity  of paths
    dir_out = sanity_util.safe_dir_path(dir_path=dir_out)
    file_name = sanity_util.unique_file_name(dir=dir_out, fn=name, suffix='.{}'.format(format))
    # reading JSON files
    p_files = sorted([os.path.join(dir_jsons, p_json) for p_json in os.listdir(dir_jsons)])
    table = defaultdict(list)
    keys = set()
    for p_f in p_files:
        if p_f.lower().endswith('.json'):
            with open(p_f, 'r') as f_json:
                el = json.load(f_json)
                for k in el.keys():
                    keys.add(k)
    for p_f in p_files:
        if p_f.lower().endswith('.json'):
            with open(p_f, 'r') as f_json:
                el = json.load(f_json)
                for k in el.keys():
                    table[k].append(el[k])
                for k in keys.difference(set(el.keys())):
                    table[k].append(None)
    # DataFrame conversion
    df = pd.DataFrame.from_dict(table)
    # writing HTML table
    if format == 'html':
        table_str = df.to_html()
    else:
        table_str = df.to_latex()
    table_str += '<script type="text/javascript" src="stylize.js"></script>'
    stylize_js = js_stylize()
    with open(os.path.join(dir_out, 'stylize.js'), 'w') as f_js:
        f_js.write(stylize_js)
    with open(file_name, 'w') as f_out:
        f_out.write(table_str)


def js_stylize():
    return '''
        /**
         * small script to stylize raw html tables
         * @author Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
         */
        
        
        /**
         * adding all bootstrap relevent dependencies to the headder
         */
        function add_bootsrap(){
            document.head.innerHTML +=
                "<link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css\">\n" +
                "<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js\"></script>\n" +
                "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js\"></script>\n" +
                "<script src=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js\"></script>";
        }
        
        
        /**
         * setting classnames of a specific tag
         */
        function style_tag(tagName, className){
            tags = document.getElementsByTagName(tagName);
            for(let i=0; i<tags.length; ++i){
                tags[i].className = className;
            }
        }
        
        
        /**
         * setting the (Bootstrap) contenteditable flag for a specific tag
         */
        function editable_tag(tagName, editable){
            tags = document.getElementsByTagName(tagName);
            for(let i=0; i<tags.length; ++i){
                tags[i].setAttribute('contenteditable', editable);
            }
        }
        
        
        // setting title
        document.title = 'PHOCNet Table';
        // adding bootstrap
        add_bootsrap();
        // stylize tables
        style_tag('table', 'table table-responsive-md');
        style_tag('thead', 'thead-dark');
        // enable editable table-divisions
        editable_tag('td', 'true'); 
    '''


def parser():
    """
    Creates a parser of this script.

    :return: args-parser with the following arguments


        Positional:

        =============== ======================================================
        arg             semantic
        =============== ======================================================
        dir_jsons       directory of JSON files
        dir_out         the directory to safe the HTML page to
        file_name       name of the HTML file
        =============== ======================================================
    """
    parser = ArgumentParser()
    parser.add_argument('dir_jsons', help='dir containing json files')
    parser.add_argument('dir_out', help='output directory')
    parser.add_argument('file_name', help='name of HTML file')
    return parser


if __name__ == '__main__':
    arg_parser = parser()
    args = vars(arg_parser.parse_args())
    jsons_to_table(dir_jsons=args['dir_jsons'], dir_out=args['dir_out'], name=args['name'], format='html')
