import os
import shutil

def move_og_image(og_image_path, safe=False):
    new_folder = og_image_path.parents[0].joinpath("original_image")
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    new_location = new_folder.joinpath(og_image_path.name)
    if os.path.exists(new_location):
        if safe:
            print('file already found at this location')
            return
        else:
            os.remove(new_location)
    shutil.move(og_image_path, new_location)


def pathcrawler(inpath, inset=set(), inlist=[], mykey=None):
    with os.scandir(inpath) as entries:
        for entry in entries:
            if os.path.isdir(entry.path) and not entry.path in inset:
                inset.add(entry.path)
                pathcrawler(entry.path, inset, inlist, mykey)
            if mykey in entry.name and os.path.isdir(entry.path):
                inlist.append(entry.path)
    return inlist
