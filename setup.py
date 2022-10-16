from setuptools import setup

setup(
    name='gpc',
    version='0.1',
    description='Generalised Partial Correlation',
    url='https://github.com/Richard-Sti/gpc',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0',
    packages=['gpc'],
    install_requires=['scipy',
                      'numpy',
                      'scikit-learn',
                      'tqdm'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9']
)
