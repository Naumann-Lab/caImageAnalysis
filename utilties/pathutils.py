import os


def pathcrawler(inpath, inset=set(), inlist=[], mykey=None):
    with os.scandir(inpath) as entries:
        for entry in entries:
            if os.path.isdir(entry.path) and not entry.path in inset:
                inset.add(entry.path)
                pathcrawler(entry.path, inset, inlist, mykey)
            if mykey in entry.name and os.path.isdir(entry.path):
                inlist.append(entry.path)
    return inlist
