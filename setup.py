from distutils.core import setup
setup(
  name = 'binomialRF',         # How you named your package folder (MyLib)
  packages = ['binomialRF'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'The binomialRF package is a random forest-based feature selection package that provides a feature selection algorithm to be used in randomForest classifiers.',   # Give a short description about your library
  author = 'Samir Rachid Zaim',                   # Type in your name
  author_email = 'samir.rachid.zaim@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/SamirRachidZaim/binomialRF.py',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['binomialRF', 'random forest', 'feature selection','decision trees'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'scikit-learn',
          'pandas',
          'statsmodels',
          'rpy2',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)