import csv

def export_rows(rows, stream):
    writer = csv.writer(stream)
    writer.writerows(rows)
