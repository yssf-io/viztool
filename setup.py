from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='VizTool',  # Required
    version='1.0',  # Required
    description='To compute and visualize features on timeseries',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='wip',  # Optional
    author='Thami EZZAIM, Youssef EL SAADANY',  # Optional
    author_email='youssef.el-saadany@unchartech.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='time series visualization',  # Optional
    package_dir={'': 'src'},
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    python_requires='>=3.7, <4',
    install_requires=['pandas',
                        'numpy',],  # Optional
    extras_require={  # Optional
        
    },
    package_data={  # Optional
        
    },
    data_files=[('my_data', ['data/data_file'])],  # Optional
    entry_points={  # Optional
        
    },
    project_urls={  # Optional
        'Bug Reports': 'wip',
        'Funding': 'Un café à midi',
        'Source': 'wip',
    },
)
