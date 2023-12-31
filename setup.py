from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
    
with open('README.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name = 'rfphate',
    version = '0.1.1',
    description = 'The package is used to generate RF-PHATE manifold embeddings based on RF-GAP or other RF proximities',
    author = 'Jake Rhodes',
    author_email = 'jakerhodes8@gmail.com',
    packages = find_packages(),
    install_requires = parse_requirements('requirements.txt'),
    package_data = {'rfphate': ['datasets/*.csv']},
    long_description = long_description
)
