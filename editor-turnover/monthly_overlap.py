import argparse
import csv
import gzip
import os

from mw.xml_dump import Iterator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file',
                        help="gzipped XML dump file -- e.g., enwiki-20190301-stub-meta-history.xml.gz")
    parser.add_argument('--outdir',
                        help="directory to write monthly editor files")
    parser.add_argument('--botlist_fn',
                        help="Text file containing bot accounts")
    parser.add_argument('--mys', nargs="+",
                        help="List of months to track of form '2016-11'")
    parser.add_argument('--startdate', default='2001-08',
                        help="If not mys, starting month to track of form '2016-11'")
    parser.add_argument('--enddate', default='2019-03',
                        help="If not mys, ending month to track of form '2016-11'")
    parser.add_argument('--stopafter', type=int, default=-1,
                        help="If greater than 0, limit to # of pages to check before stopping")
    args = parser.parse_args()

    # build list of months to track
    if not args.mys:
        args.mys = []
        sd = (int(args.startdate[:4]), int(args.startdate[5:]))
        ed = (int(args.enddate[:4]), int(args.enddate[5:]))
        while ed[0] > sd[0] or ed[1] >= sd[1]:
            args.mys.append('{0}-{1:02}'.format(sd[0], sd[1]))
            if sd[1] == 12:
                sd = (sd[0]+1, 1)
            else:
                sd = (sd[0], sd[1]+1)

    print(args)

    # load in bot usernames
    bots = set()
    if args.botlist_fn:
        with open(args.botlist_fn) as fin:
            csvreader = csv.reader(fin)
            for line in csvreader:
                bots.add(line[0].lower())

    # Construct dump file iterator
    dump = Iterator.from_file(gzip.open(args.dump_file))

    editors_by_month = {}
    editor_startdates = {}
    i = 0
    bots_edits_filtered = 0
    user_edits = 0
    anon_edits = 0
    print_every = 25
    # Iterate through pages
    for page in dump:

        # only count edits to article namespace
        if page.namespace != 0:
            continue
        i += 1

        # Iterate through a page's revisions
        for revision in page:
            contributor = revision.contributor
            if not contributor.id or contributor.id == 0:
                anon_edits += 1
                continue
            editor_name = contributor.user_text
            if editor_name.lower() in bots:
                bots_edits_filtered += 1
                continue
            month_year = revision.timestamp.strftime("%Y-%m")
            editor_startdates[editor_name] = min(month_year, editor_startdates.get(editor_name, "2100-01"))
            if args.mys and month_year not in args.mys:
                continue
            user_edits += 1
            if month_year not in editors_by_month:
                editors_by_month[month_year] = {}
            editors_by_month[month_year][editor_name] = editors_by_month[month_year].get(editor_name, 0) + 1

        if i == args.stopafter:
            break

        if i % print_every == 0:
            print('{0} completed. On: {1}. {2} bot edits, {3} anon edits, {4} user edits.'.format(
                i, page.title, bots_edits_filtered, anon_edits, user_edits))
            print_every *= 2

    print('{0} completed. On: {1}. {2} bot edits, {3} anon edits, {4} user edits.'.format(
        i, page.title, bots_edits_filtered, anon_edits, user_edits))

    for my in editors_by_month:
        fn = os.path.join(args.outdir, my)
        with open(fn, 'w') as fout:
            csvwriter = csv.writer(fout)
            csvwriter.writerow(['editor_name', 'edit_count', 'first_edit_dt'])
            d = editors_by_month[my]
            by_editcount = [(k, d[k], editor_startdates[k]) for k in sorted(d, key=d.get, reverse=True)]
            for editor_count in by_editcount:
                csvwriter.writerow(editor_count)




if __name__ == "__main__":
    main()