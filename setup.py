from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='TetraPredX',
      packages=['TetraPredX'],
      description='Prediction tool to identify the class of unknown sequences assembled from metagenomes',
      long_description=readme(),
      long_description_content_type='text/markdown',
      version='1.0',
      author='Sejal Modha',
      author_email='s.modha.1@research.gla.ac.uk',
      url='https://github.com/sejmodha/TetraPredX',
      download_url='https://github.com/sejmodha/TetraPredX/archive/refs/tags/1.0.tar.gz',
      install_requires=['biopython', 
                        'joblib', 
                        'sklearn=0.24.1',
                        'pathos', 
                        'seaborn'],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Intended Audience :: Science/Research',
    	  'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      python_requires="~=3.6",
      keywords='bioinformatics tetranucleotides microbes',
      
     )
