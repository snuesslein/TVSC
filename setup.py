from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='tvsclib',
    packages=find_packages(include=['tvsclib']),
    version='0.1.0',
    description='Time varying systems computation library',
    author='ge37bov@mytum.de',
    license='MIT',
    install_requires=required,
    setup_requires=['setuptools-lint',
                    'Sphinx',
                    'sphinx-rtd-theme',
                    'sphinxcontrib-apidoc'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)