from setuptools import find_packages, setup 
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path:str) ->List[str]:
    requirements= []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

__version__ = "0.0.1"
SRC_REPO = "churn"

setup(
    name=SRC_REPO,
    version = __version__,
    license = 'MIT',
    url = 'https://github.com/minich-code/churnpred.git',
    download_url = f'https://github.com/minich-code/churnpred',
    description='This is a project to Classify customer churn',
    author = 'Western',
    author_email = 'minichworks@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
