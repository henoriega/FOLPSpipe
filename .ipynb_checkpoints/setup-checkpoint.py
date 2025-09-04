import os
import sys
from setuptools import setup, find_packages

package_basename = 'folps'
package_dir = os.path.join(os.path.dirname(__file__), package_basename)
sys.path.insert(0, package_dir)
import _version
version = _version.__version__

# Read long description from README.md
with open('README.md', encoding='utf-8') as f:
      long_description = f.read()

setup(
      name=package_basename,
      version=version,
      author='Hernan E. Noriega',
      author_email='',
      description='FOLPS code (includes JAX)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='BSD3',
      url='https://github.com/henoriega/FOLPSpipe',
      install_requires=['numpy', 'scipy', 'jax', 'interpax'],
      extras_require={},
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.7',
      classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
      ],
)
