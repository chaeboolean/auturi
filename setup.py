from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

print("find_packages", find_packages("auturi"))
setup(
   name="auturi",
   version="0.0",
   description="Experiment source code",
   author="Chaehyun Jeong",
   author_email="chjeong9727@snu.ac.kr",
   packages=["auturi"],
)