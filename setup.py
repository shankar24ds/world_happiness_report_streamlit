from setuptools import setup, find_packages

setup(
    name='Word Happiness Report in Streamlit',
    version='0.1.0',
    author='Shankar S',
    author_email='shankarselvaraj24@gmail.com',
    url='https://github.com/shankar24ds/world_happiness_report_streamlit',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'pandas==1.5.3',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'jupyter==1.0.0',
        'streamlit==1.25.0',
        'plotly==5.9.0'
    ],
)