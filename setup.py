from setuptools import find_packages, setup

setup(
    name='tvsclib',
    packages=find_packages(include=['tvsclib']),
    version='0.1.0',
    description='Time varying systems computation library',
    author='ge37bov@mytum.de',
    license='MIT',
    install_requires=['numpy','scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)