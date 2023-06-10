from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

with open('LICENSE.txt') as f:
    LICENSE = f.read()

setup_args = dict(
    name='counterplots',
    version='0.0.3',
    description='Plotting tool for counterfactual explanations',
    long_description_content_type='text/markdown',
    long_description=README,
    license='MIT',
    packages=find_packages(exclude=('tests\*', 'exp_notebooks\*', '_static\*')),
    author='Raphael Mazzine Barbosa de Oliveira, Bjorge Meulemeester',
    keywords=['Counterfactual Explanations', 'Visualization', 'Plotting', 'Explainable Artificial Intelligence', 'XAI', 'Machine Learning'],
    url='https://github.com/ADMAntwerp/CounterPlots',
    download_url='https://pypi.org/project/counterplots/',
)

install_requires = [
    'numpy',
    'matplotlib',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
