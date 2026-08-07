from io import StringIO
from src.export import export_rows

def test_export():
    s=StringIO(); export_rows([['a','b']],s); assert s.getvalue()
