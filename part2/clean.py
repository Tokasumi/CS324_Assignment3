from pathlib import Path
import shutil


def clean():
    _dir = 'images'
    shutil.rmtree(_dir)
    Path(_dir).mkdir()
