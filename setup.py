from setuptools import find_packages, setup


with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

name = "thunder-examples"

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version="0.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
)
