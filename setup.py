from setuptools import find_packages, setup

setup(
    name="datatuner",
    version="1.0",
    description="Natural Language Generation Library",
    packages=find_packages(),
    package_dir={"": "src"},
    package_data={},
    install_requires=[],
    extras_require={},
    zip_safe=False,
    tests_require=[],
)
