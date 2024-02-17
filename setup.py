from setuptools import setup, find_packages

setup(
    name="zen_garden",
    version="0.1",
    description="A Sentence about the package.",
    author="Foo Bar, Spam Eggs",
    author_email="foobar@baz.com, spameggs@joe.org",
    python_requires=">=3.8, <4",
    keywords="key1, key2",
    packages=find_packages(),
    project_urls={"ZEN-Garden": "https://github.com/RRE-ETH/ZEN-garden"},
    install_requires=[
        "cython",
        "numpy",
        "pandas",
        "geopandas",
        "scipy",
        "coverage",
        "pint",
        "matplotlib",
        "entsoe-py",
        "h5py",
        "tables",
        "ordered-set",
        "linopy",
        "psutil",
        "tsam"
    ],
)
