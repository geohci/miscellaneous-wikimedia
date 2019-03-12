import argparse
from datetime import datetime
import csv
import os

def fn_to_dt(fn):
    dt_str = os.path.basename(fn)
    return datetime.strptime(dt_str, "%Y-%m")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monthly_fns', nargs="+", help="CSV files with monthly editors")
    args = parser.parse_args()

    if len(args.monthly_fns) == 1:
        dir = args.monthly_fns[0]
        args.monthly_fns = sorted([os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
    else:
        args.monthly_fns = sorted(args.monthly_fns)

    editors_by_month = {}
    editor_starts = {}
    for fn in args.monthly_fns:
        print("Processing", fn)
        editors_by_month[fn] = {}
        with open(fn, 'r') as fin:
            csvreader = csv.reader(fin)
            header = next(csvreader)
            assert header == ['editor_name', 'edit_count', 'first_edit_dt']
            for line in csvreader:
                name = line[0]
                count = int(line[1])
                first_edit = line[2]
                editors_by_month[fn][name] = count
                editor_starts[name] = datetime.strptime(first_edit, "%Y-%m")

    # stats for editors / edits by whether they are made by new or old editors
    print("Date    \tTotal Editors\tNew\tOld\t\tTotal Edits\tNew\tOld")
    for fn in args.monthly_fns:
        new_edits = {}
        old_edits = {}
        dt = fn_to_dt(fn)
        for e in editors_by_month[fn]:
            count_edits = editors_by_month[fn][e]
            if editor_starts[e] == dt:
                new_edits[e] = count_edits
            else:
                old_edits[e] = count_edits
        count_new_edits = sum(new_edits.values())
        count_new_editors = len(new_edits)
        count_old_edits = sum(old_edits.values())
        count_old_editors = len(old_edits)
        total_edits = count_new_edits + count_old_edits
        total_editors = count_new_editors + count_old_editors
        print("{0}:\t{1}\t{2:.2f}\t{3:.2f}\t\t{4}\t{5:.2f}\t{6:.2f}".format(dt,
                                                                            total_editors,
                                                                            count_new_editors / total_editors,
                                                                            count_old_editors / total_editors,
                                                                            total_edits,
                                                                            count_new_edits / total_edits,
                                                                            count_old_edits / total_edits))
    # stats for what proportion of editors / edits in a month are new / old
    print("\n==========")
    start_month = args.monthly_fns[0]
    start_dt = fn_to_dt(start_month)
    start_editors = editors_by_month[start_month].keys()
    print("{0}:\t{1} editors".format(os.path.basename(start_month), len(start_editors)))
    for i in range(1, len(args.monthly_fns)):
        compare_month = args.monthly_fns[i]
        compare_editors = editors_by_month[compare_month].keys()
        overlap = list(start_editors & compare_editors)
        start_since = 0
        edits_since = 0
        edits_common = 0
        total_edits = sum(editors_by_month[compare_month].values())
        editors_common = len(overlap)
        total_editors = len(compare_editors)
        for e in compare_editors:
            num_edits = editors_by_month[compare_month][e]
            if editor_starts[e] > start_dt:
                start_since += 1
                edits_since += num_edits
            elif e in overlap:
                edits_common += num_edits
        print("{0}:\t{1} editors;\t{2} ({3:.2f}) in common;\t{4} ({5:.2f}) new since;"
              "\t{6} edits;\t{7} ({8:.2f}) in common;\t{9} ({10:.2f}) new since".format(
            os.path.basename(compare_month),
            total_editors,
            editors_common, editors_common / total_editors,
            start_since, start_since / total_editors,
            total_edits,
            edits_common, edits_common / total_edits,
            edits_since, edits_since / total_edits))


if __name__ == "__main__":
    main()