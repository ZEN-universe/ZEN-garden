from setuptools import setup

setup(
    name="zen_garden",
    version="0.1",
    description='A Sentence about the package.',
    author='Foo Bar, Spam Eggs',
    author_email='foobar@baz.com, spameggs@joe.org',
    python_requires='>=3.8, <4',
    keywords='key1, key2',
    packages=['zen_garden'],
    package_dir={'zen_garden': 'zen_garden/'},
    project_urls={'ZEN-Garden': 'https://github.com/RRE-ETH/ZEN-garden'},
)
