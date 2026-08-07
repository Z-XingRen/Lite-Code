from src.csv_ingest import read_records

def test_ingest():
    assert read_records('name,score\nA,1\n')[0]['score'] == 1.0
