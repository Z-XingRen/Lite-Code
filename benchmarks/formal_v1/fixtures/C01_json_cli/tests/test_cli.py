import subprocess,sys

def test_cli():
    p=subprocess.run([sys.executable,'src/cli.py'],input='{"name":"a","value":2}\n',text=True,capture_output=True)
    assert p.returncode == 0
