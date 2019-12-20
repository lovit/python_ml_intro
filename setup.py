import mydata
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requires = f.read().splitlines()

setup(
    name="mydata",
    version=mydata.__version__,
    author=mydata.__author__,
    author_email='soy.lovit@gmail.com',
    url='https://github.com/lovit/python_ml_intro',
    description="Introduction of machine learning for data analytics",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requires,
    packages=find_packages(),
    package_data={
        'mydata':['data/datafile/*']
    },
)
