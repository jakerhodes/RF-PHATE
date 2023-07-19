from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
    
    
setup(
    name = 'RFPHATE',
    version = '0.1.0',
    description = 'The package is used to generate RF-PHATE manifold embeddings based on RF-GAP or other RF proximities',
    author = 'Jake Rhodes',
    author_email = 'jakerhodes8@gmail.com',
    packages = find_packages(),
    install_requires = parse_requirements('requirements.txt')
)
