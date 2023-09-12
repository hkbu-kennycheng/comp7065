from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
import gzip
import json

domain = Domain([
    ContinuousVariable("overall"),
    ContinuousVariable("verified"),
    ContinuousVariable("unixReviewTime")
], metas=[
    StringVariable("reviewTime"),
    StringVariable("reviewerID"),
    StringVariable("asin"),
    StringVariable("style"),
    StringVariable("reviewerName"),
    StringVariable("reviewText"),
    StringVariable("summary")
])

arr = []
meta = []

with gzip.open('Magazine_Subscriptions.json.gz', 'rb') as f:
    for line in f:
        record = json.loads(line)
        if record['verified']:
            record['verified'] = 1
        else:
            record['verified'] = -1

        if 'style' in record.keys():
            record['style'] = record['style']['Format:']
        else:
            record['style'] = 'None'

        if 'reviewText' not in record.keys():
            record['reviewText'] = ''
        if 'summary' not in record.keys():
            record['summary'] = ''
        if 'reviewerName' not in record.keys():
            record['reviewerName'] = ''

        arr.append([record['overall'], record['verified'], record['unixReviewTime']])
        meta.append([record['reviewTime'], record['reviewerID'], record['asin'], record['style'], record['reviewerName'], record['reviewText'], record['summary']])

out_data = Table.from_numpy(domain, arr, metas=meta)
