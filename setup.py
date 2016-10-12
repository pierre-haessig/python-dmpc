from setuptools import setup

setup(

    name = "dmpc",
    version = '0.1',
    packages = ['dmpc', 'dmpc.tests'],

    test_suite='dmpc.tests',
    install_requires = ["numpy", "cvxopt"],

    extras_require = {
            'plots':  ["matplotlib"],
    },

    url='https://github.com/pierre-haessig/python-dmpc',
    author = "Pierre Haessig",
    author_email = "pierre.haessig@centralesupelec.fr",

    description = 'simulation tool for Model Predictive Control (MPC) and Distributed MPC',
    long_description = '',

    license = 'BSD-3',

)
