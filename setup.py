import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bayesfem',
    version='0.0.1',
    author='Daniel Tait',
    author_email='tait.djk@gmail.com',
    description='Bayes inf. + FEM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/danieljtait/bayesfem',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
