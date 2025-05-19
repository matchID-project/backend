import importlib.util
from pathlib import Path
import pytest

pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')

MODULE_PATH = Path(__file__).resolve().parents[1] / 'code' / 'recipes.py'

def load_recipes():
    try:
        spec = importlib.util.spec_from_file_location('recipes', MODULE_PATH)
        recipes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipes)
        return recipes
    except Exception as e:
        pytest.skip(f'Could not import recipes module: {e}')

recipes = load_recipes()


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
