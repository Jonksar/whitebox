from google_images_download.google_images_download import googleimagesdownload
import os, sys
import shutil
import json
import sqlite3
from uuid import uuid4, UUID
from contextlib import suppress
from datetime import datetime
from pprint import pprint

from glob import glob

sqlite3.register_adapter(UUID, lambda u: u.bytes_le)
sqlite3.register_converter('GUID', lambda b: UUID(bytes_le=b))


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


class DatasetDatabase:
    def __init__(self, path):
        self._path = path
        self.conn = sqlite3.connect(self._path)
        self.c = self.conn.cursor()

        # Create table
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS annotations
                         (annotation GUID PRIMARY KEY, 
                          object GUID,
                          dataset GUID,
                          instruction_version int,
                          result integer,
                          payload text,
                          created_at text,
                          updated_at text, 
                          deleted_at text
                          )
        ''')

        # Create table for all objects for annotations
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS objects
                         (
                          object GUID PRIMARY KEY,
                          dataset GUID,
                          path text,
                          status text,
                          created_at text,
                          updated_at text, 
                          deleted_at text
                          )
        ''')

    def insert_new_object(self,
                          path,  # Path to file
                          dataset,  # Dataset name
                          status='waiting_annotations'):
        _object = uuid4()
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        updated_at = created_at
        deleted_at = ""

        self.c.execute('''
                     INSERT INTO objects (object, dataset, path, status, created_at, updated_at, deleted_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)
                 ''', (
            _object, dataset, path, status, created_at, updated_at, deleted_at
        ))

    def insert_new_annotation(self,
                              _object,
                              dataset,
                              result=True,
                              payload={},
                              instruction_version=0,
                              new_status='completed'):
        annotation = uuid4()
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        updated_at = created_at
        deleted_at = ""

        self.c.execute('''
                     INSERT INTO annotations (annotation, object, dataset,
                                    instruction_version, result, payload,
                                 created_at, updated_at, deleted_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ''', (
            annotation, _object, dataset,
            instruction_version, int(result),
            json.dumps(payload), created_at, updated_at, deleted_at
        ))

        self.c.execute('''
                     UPDATE objects SET status = ?
                     WHERE object = ? AND dataset = ?;

                 ''', (new_status, _object, dataset)
                       )

    def get_completed_annotations(self, dataset=None):
        results = self.c.execute("""SELECT objects.object, objects.path, annotations.result, annotations.payload
        FROM objects
        JOIN annotations on objects.object = annotations.object
        WHERE objects.dataset =? AND objects.status =?""",
                                 (dataset, 'completed')).fetchall()

        return results

    def get_waiting_annotations(self, dataset=None):
        results = self.c.execute("SELECT object, path FROM objects WHERE dataset=? AND status=?",
                                 (dataset, 'waiting_annotations')).fetchall()
        return results

class Dataset:
    _data_dir = os.path.expanduser('~/.whitebox')

    def __init__(self, _id=None):
        self._uuid = str(uuid4()) if _id is None else _id
        self._dataset_dir = os.path.join(self._data_dir, self._uuid)

        # Create metadata database
        self._database_file = os.path.join(self._data_dir, "annotations.db")
        self.database = DatasetDatabase(self._database_file)

    def dataset_info(self):
        assert self._uuid is not None, "Need to have self._id at initialization"
        classes = os.listdir(self._dataset_dir)

        result = {c:
                      [(self._dataset_dir + '/' + c + '/' + path) for path in
                       os.listdir(os.path.join(self._dataset_dir, c))]
                  for c in classes}

        return result

    def get_waiting_annotations(self):
        return self.database.get_waiting_annotations(self._uuid)

    def get_completed_annotations(self):
        return self.database.get_completed_annotations(self._uuid)

    def insert_new_annotation(self, _object, result=True, payload={}, instruction_version=0, new_status='completed'):
        return self.database.insert_new_annotation(_object, dataset=self._uuid,
                                                   result=result, payload=payload,
                                                   instruction_version=instruction_version,
                                                   new_status=new_status)

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

                self.database.insert_new_object(destination, self._uuid)

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
