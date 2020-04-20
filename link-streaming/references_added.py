import json
from sseclient import SSEClient as EventSource

#rc_url = 'https://stream.wikimedia.org/v2/stream/recentchange'
ref_url = 'https://stream.wikimedia.org/v2/stream/page-links-change'
for event in EventSource(ref_url):
    if event.event == 'message':
        try:
            change = json.loads(event.data)
        except ValueError:
            continue
#        print(change.keys())
        if 'added_links' in change:
            added_ext_links = [l for l in change['added_links'] if '.cn' in l['link']]
            if added_ext_links:
                print('{0} ({1}): {2}'.format(change['page_title'], change['database'], added_ext_links))