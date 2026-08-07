from src.invoice import Invoice
from src.render import render_invoice

def test_render():
    assert '12.50' in render_invoice(Invoice('I-1', 12.5))
