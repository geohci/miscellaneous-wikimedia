import argparse
import bz2
import json
from sseclient import SSEClient as EventSource

REF_URL = 'https://stream.wikimedia.org/v2/stream/page-create'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikis", nargs="*",
                        help="List of Wikipedia databases to filter by -- e.g., 'wikidatawiki enwiki eswiki' -- "
                             "otherwise no filtering done by wiki.")
    parser.add_argument("--output_json", help="Filename to JSON file where results will be saved.")
    parser.add_argument("--keyword_fn", help="Text file where each line is a keyword to match against created pages.")
    args = parser.parse_args()

    fout = None
    if args.output_json:
        if args.output_json.endswith('.bz2'):
            fout = bz2.open(args.output_json, 'wt')
        else:
            fout = open(args.output_json, 'w')

    keywords = ['file']
    if args.keyword_fn:
        keywords = []
        with open(args.keyword_fn, 'r') as fin:
            for line in fin:
                keywords.append(line.strip().lower().replace(' ', '_'))

    for i,event in enumerate(EventSource(REF_URL)):
        if event.event == 'message':
            try:
                change = json.loads(event.data)
            except ValueError:
                continue
            if not args.wikis or change['database'] in args.wikis:
                title = change['page_title'].lower()
                for k in keywords:
                    if k in title:
                        print(i, change)
                        if fout is not None:
                            fout.write(json.dumps(change) + '\n')
                        break

        if i % 10000 == 0:
            print("{0} pages have been created. {1} met filter criteria.")

if __name__ == "__main__":
    main()