from setuptools import setup

setup(
    name='tensorflow-ultrasound',
    version='0.1.0',
    description='A Tensorflow-dependent package for Ultrasound scan convert process',
    url='https://github.com/stevencheng1220/tensorflow-ultrasound',
    author='Steven Cheng, Ouwen Huang',
    author_email='sc618@duke.edu',
    license='Apache License',
    packages=['tensorflow-ultrasound'],
    install_requires=['tensorflow',
                      'tensorflow_addons'
                     ],
    
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License',  
        'Programming Language :: Python :: 3',
    ],
)