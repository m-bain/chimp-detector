__author__      = 'Ernesto Coto'
__copyright__   = 'April 2019'

import zipfile
import os

def zipdir(filename, path_to_compress):
    """
        Compresses a folder to a ZIP file.
        Arguments:
            filename: output ZIP filename
            path_to_compress: source folder for the compression
    """
    zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path_to_compress):
        for afile in files:
            zipf.write(os.path.join(root, afile))
    # close zip
    zipf.close()


def remove_unused_images(path, used_files):
    """
        Removes a set of files from a folder
        Arguments:
            path: source folder containing images
            used_files: dictionary of filenames lists, grouped by track-ids
    """
    for im_fname in os.listdir(path):
        full_path = os.path.join(path, im_fname)
        # do a brute-force search for the file in used_files
        frame_found = False
        for track_id in used_files.keys():
            if im_fname in used_files[track_id]:
                frame_found = True
                break
        if not frame_found and im_fname.endswith('.jpg'):
            os.remove(full_path)

