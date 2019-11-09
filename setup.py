import lovit_analytics_introduction
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requires = f.read().splitlines()

setup(
    name="lovit_analytics_introduction",
    version=lovit_analytics_introduction.__version__,
    author=lovit_analytics_introduction.__author__,
    author_email='soy.lovit@gmail.com',
    url='https://github.com/lovit/python_ml_intro',
    description="Introduction of machine learning for data analytics",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requires,
    packages=find_packages(),
    package_data={
        'lovit_analytics_introduction':[]
    },
)
