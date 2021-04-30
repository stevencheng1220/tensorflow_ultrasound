from setuptools import setup

setup(
    name='tensorflow_ultrasound',
    version='0.1.5',
    description='A Tensorflow-dependent package for Ultrasound scan convert process',
    url='https://github.com/stevencheng1220/tensorflow_ultrasound',
    author='Steven Cheng, Ouwen Huang',
    author_email='sc618@duke.edu',
    license='Apache License 2.0',
    packages=['tensorflow_ultrasound'],
    install_requires=['tensorflow',
                      'tensorflow_addons'
                     ],
    
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)