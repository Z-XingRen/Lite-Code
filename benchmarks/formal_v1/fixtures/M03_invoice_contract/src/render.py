from .invoice import Invoice

def render_invoice(invoice: Invoice):
    return f'{invoice.number}: {invoice.total:.2f}'
