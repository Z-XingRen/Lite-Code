import csv
from io import StringIO

def read_records(text):
    rows = csv.DictReader(StringIO(text))
    return [{'name': row['name'], 'score': float(row['score'])} for row in rows]
