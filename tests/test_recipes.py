# -*- coding: utf-8 -*-
import importlib.util
from pathlib import Path
import sys
import os
import pytest

pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')

# Chargement dynamique du module recipes
MODULE_PATH = Path(__file__).resolve().parents[1] / 'code' / 'recipes.py'
CODE_DIR = str(MODULE_PATH.parent)

def load_module(name, path):
    original_sys_path = list(sys.path)
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Impossible de charger {name} depuis {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = original_sys_path

recipes = load_module('recipes', MODULE_PATH)
config_path = Path(__file__).resolve().parents[1] / 'code' / 'config.py'
config = load_module('config', config_path)

#############################
# Tests                     #
#############################

def test_fwf_format_and_to_fwf(tmp_path):
    df = pd.DataFrame({'A': ['a', 'bb'], 'B': ['1', '22']})
    widths = [3, 3]
    line = recipes.fwf_format(df.iloc[0], widths)
    assert line == 'a  1  '
    outfile = tmp_path / 'out.txt'
    recipes.to_fwf(df, outfile, widths=widths, names=['A', 'B'])
    content = outfile.read_text().splitlines()
    assert content[0].strip() == line.strip()


def test_internal_fillna_and_keep():
    df = pd.DataFrame({'A': [1, None], 'B': [None, 'x']})
    r_fill = recipes.Recipe.__new__(recipes.Recipe)
    r_fill.args = [{'A': 0, 'B': ''}]
    filled = recipes.Recipe.internal_fillna(r_fill, df.copy())
    assert filled['A'].tolist() == [1, 0]
    assert filled['B'].tolist() == ['', 'x']

    r_keep = recipes.Recipe.__new__(recipes.Recipe)
    r_keep.args = {'select': ['A']}
    recipes.Recipe.select_columns(r_keep, filled)
    kept = recipes.Recipe.internal_keep(r_keep, filled)
    assert list(kept.columns) == ['A']


def test_internal_rename():
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    r = recipes.Recipe.__new__(recipes.Recipe)
    r.args = {'new_A': 'A', 'new_B': 'B'}
    renamed = recipes.Recipe.internal_rename(r, df.copy())
    assert list(renamed.columns) == ['new_A', 'new_B']


def test_internal_map():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    r = recipes.Recipe.__new__(recipes.Recipe)
    r.args = {'C': 'A', 'D': ['A', 'B']}
    mapped = recipes.Recipe.internal_map(r, df.copy())
    assert mapped['C'].tolist() == [1, 2, 3]
    assert mapped['D'].tolist() == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_internal_shuffle():
    df = pd.DataFrame({'A': range(10), 'B': list('abcdefghij')})
    r = recipes.Recipe.__new__(recipes.Recipe)
    shuffled = recipes.Recipe.internal_shuffle(r, df.copy())
    assert set(shuffled['A']) == set(df['A'])
    assert set(shuffled['B']) == set(df['B'])


# Utilitaire pour créer un objet Recipe minimal avec un logger muet
def _recipe_with_args(args):
    r = recipes.Recipe.__new__(recipes.Recipe)
    r.args = args
    r.log = type('log', (), {'write': lambda self, *a, **k: None})()
    return r


def test_internal_to_integer_single():
    df = pd.DataFrame({'A': ['1', '2', '', '']})
    r = _recipe_with_args({'select': ['A']})
    converted = recipes.Recipe.internal_to_integer(r, df.copy())
    assert converted['A'].tolist()[:2] == [1, 2]
    assert pd.isna(converted['A'].iloc[2]) and pd.isna(converted['A'].iloc[3])


def test_internal_to_float():
    df = pd.DataFrame({'A': ['1.1', '', '']})
    r = _recipe_with_args({'select': ['A']})
    converted = recipes.Recipe.internal_to_float(r, df.copy())
    assert converted['A'].iloc[0] == 1.1
    assert pd.isna(converted['A'].iloc[1]) and pd.isna(converted['A'].iloc[2])


def test_internal_parsedate():
    df = pd.DataFrame({'A': ['2023-01-01', '2023-02-01']})
    r = _recipe_with_args({'select': ['A'], 'format': '%Y-%m-%d'})
    parsed = recipes.Recipe.internal_parsedate(r, df.copy())
    assert pd.api.types.is_datetime64_any_dtype(parsed['A'])


def test_internal_normalize():
    df = pd.DataFrame({'A': ['été', 'naïve']})
    r = recipes.Recipe.__new__(recipes.Recipe)
    r.args = {'select': ['A']}
    normalized = recipes.Recipe.internal_normalize(r, df.copy())
    assert normalized['A'].tolist() == ['ete', 'naive']


def test_internal_pause():
    df = pd.DataFrame({'A': [1, 2]})
    r = recipes.Recipe.__new__(recipes.Recipe)
    paused = recipes.Recipe.internal_pause(r, df.copy())
    assert paused.equals(df)


def test_internal_list_tuple():
    df = pd.DataFrame({'A': [[1, 2], [3, 4]]})
    r1 = recipes.Recipe.__new__(recipes.Recipe)
    r1.args = {'select': ['A']}
    tuples = recipes.Recipe.internal_list_to_tuple(r1, df.copy())
    assert all(isinstance(x, tuple) for x in tuples['A'])

    r2 = recipes.Recipe.__new__(recipes.Recipe)
    r2.args = {'select': ['A']}
    lists = recipes.Recipe.internal_tuple_to_list(r2, tuples.copy())
    assert all(isinstance(x, list) for x in lists['A'])


def test_internal_sql():
    df = pd.DataFrame({'A': [1]})
    r = recipes.Recipe.__new__(recipes.Recipe)

    # Mock minimal pour input.connector.sql
    class DummySQL:
        def execute(self, query):
            return None
    r.input = type('inp', (), {'connector': type('conn', (), {'sql': DummySQL()})()})

    r.args = "SELECT 1"
    assert recipes.Recipe.internal_sql(r, df.copy()).equals(df)


def test_internal_unnest_and_nest():
    df = pd.DataFrame({'A': [[1, 2], [3, 4]], 'B': ['x', 'y']})
    r_un = recipes.Recipe.__new__(recipes.Recipe)
    r_un.args = {'select': ['A']}
    unnest = recipes.Recipe.internal_unnest(r_un, df.copy())
    assert len(unnest) == 2
    assert 'A' not in unnest.columns or isinstance(unnest.iloc[0]['A'], list)

    df2 = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    r_nest = recipes.Recipe.__new__(recipes.Recipe)
    r_nest.args = {'select': ['A', 'B'], 'target': 'nested'}
    nested = recipes.Recipe.internal_nest(r_nest, df2.copy())
    assert 'nested' in nested.columns
    assert 'A' not in nested.columns and 'B' not in nested.columns

#############################
#  Nouveaux tests ajoutés   #
#############################

def test_internal_keep():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['x', 'y', 'z', 'w'],
        'flag': [True, False, True, False]
    })
    r = _recipe_with_args({'select': ['A', 'B'], 'where': 'flag == True'})
    kept = recipes.Recipe.internal_keep(r, df.copy())
    # Doit garder uniquement lignes avec flag==True et colonnes A,B
    assert list(kept.columns) == ['A', 'B']
    assert kept['A'].tolist() == [1, 3]


def test_internal_delete():
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y'], 'C': [10, 20]})
    r = _recipe_with_args({'select': ['B']})
    deleted = recipes.Recipe.internal_delete(r, df.copy())
    assert 'B' not in deleted.columns and list(deleted.columns) == ['A', 'C']


def test_internal_replace():
    df = pd.DataFrame({'A': ['abc123', 'def456']})
    r = _recipe_with_args({'select': ['A'], 'regex': [{'[0-9]+': 'NUM'}]})
    replaced = recipes.Recipe.internal_replace(r, df.copy())
    assert replaced['A'].tolist() == ['abcNUM', 'defNUM']


def test_internal_groupby():
    df = pd.DataFrame({'grp': ['g1', 'g1', 'g2'], 'val': [1, 2, 3]})
    r = _recipe_with_args({'select': ['grp'], 'groupby': ['grp'], 'agg': {'val': 'sum'}})
    grouped = recipes.Recipe.internal_groupby(r, df.copy())
    assert grouped['val'].tolist() == [3, 3]


def test_internal_ngram():
    df = pd.DataFrame({'txt': ['hello world']})
    # 'n' doit être une liste pour la fonction ngrams ; on utilise [2] pour les bigrammes
    r = _recipe_with_args({'select': ['txt'], 'n': [2]})
    result = recipes.Recipe.internal_ngram(r, df.copy())
    # La fonction remplace simplement la colonne sélectionnée par la liste de n-grammes
    assert 'txt' in result.columns
    # Premier bigramme attendu : 'he'
    assert result['txt'].iloc[0][0] == 'he'


def test_internal_exec():
    df = pd.DataFrame({'A': [1, 2]})
    r = _recipe_with_args("df['B'] = df['A'] * 10")
    executed = recipes.Recipe.internal_exec(r, df.copy())
    assert executed['B'].tolist() == [10, 20]


def test_internal_eval():
    df = pd.DataFrame({'A': [1, 2]})
    r = _recipe_with_args([{'B': 'A * 5'}])
    evaluated = recipes.Recipe.internal_eval(r, df.copy())
    assert evaluated['B'].tolist() == [5, 10]