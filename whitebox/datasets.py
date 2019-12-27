from google_images_download.google_images_download import googleimagesdownload
import os, sys
import shutil
from uuid import uuid4
from contextlib import suppress
from pprint import pprint

from glob import glob


def last_directory_name(path):
    return os.path.split(os.path.dirname(path))[1]


class DatasetStatistics:
    def __init__(self, name, **kwargs):
        self.name = name
        self.metadata = kwargs

    def __str__(self):
        result = f"{self.name}"
        for key, value in self.metadata.items():
            result += f"\n\t- {key}: {value}"
        result += "\n"
        return result


class Dataset:
    _data_dir = os.path.expanduser('~/.whitebox')

    def __init__(self, _id=None):
        self._uuid = str(uuid4()) if _id is None else _id
        self._dataset_dir = os.path.join(self._data_dir, self._uuid)

    def dataset_info(self):
        assert self._uuid is not None, "Need to have self._id at initialization"
        classes = os.listdir(self._dataset_dir)

        result = { c:
                [ (self._dataset_dir + '/' + c + '/' + path) for path in os.listdir(os.path.join(self._dataset_dir, c))]
                for c in classes}

        return result

    def dataset_from_paths_dict(self, paths_dict: dict, copy=False):
        # If OSError, the directory already exists
        with suppress(OSError):
            os.makedirs(self._dataset_dir)

        # select if we copy or move files
        copy = shutil.copy if copy else shutil.move

        for _class, paths in paths_dict.items():
            destination_directory = os.path.join(self._dataset_dir, _class)
            # If OSError, the directory already exists
            with suppress(OSError):
                os.makedirs(destination_directory)

            for path in paths:
                # Make destination path
                destination = os.path.join(destination_directory, os.path.basename(path))

                # Actually does the file moving / copying
                copy(path, destination)

        return self._uuid

    def list_datasets(self):
        return

    def dataset_statistics(self, dataset_uuid=None):
        data_directories = glob(os.path.join(self._data_dir, "*/"))

        for dataset_path in data_directories:
            dataset_name = last_directory_name(dataset_path)
            # if dataset uuid is specified, continue everything that is asked of
            if dataset_uuid is not None and dataset_uuid != dataset_name:
                continue

            dataset_statistics = None
            classes = glob(os.path.join(dataset_path, '*/'))

            class_counts = {}

            for _class in classes:
                files = glob(os.path.join(dataset_path, _class, '*'))
                num_files = len(files)
                class_counts[last_directory_name(_class)] = num_files

            print(DatasetStatistics(dataset_name, **class_counts))


class GoogleImageDownloader(Dataset):
    def __init__(self, _id=None):
        super().__init__(_id)
        self._downloader = googleimagesdownload()

    def download_query_images(self, query, class_name=None, limit=10):
        response = self._downloader.download({'keywords': query, 'limit': limit})
        response = response[0]
        result_paths = [path for path in response[query]]
        return result_paths

