from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
    
with open('README.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='rfphate',
    version='0.1.2',
    description='The package is used to generate RF-PHATE manifold embeddings based on RF-GAP or other RF proximities',
    author='Jake Rhodes',
    author_email='jakerhodes8@gmail.com',
    url='https://github.com/jakerhodes8/rfphate',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    package_data={'rfphate': ['datasets/*.csv']},
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
