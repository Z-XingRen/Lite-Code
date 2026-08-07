from src.storage import write_json

def test_write(tmp_path):
    p=tmp_path/'x'/'a.json'; write_json(p, {'b':1,'a':2}); assert p.exists()
