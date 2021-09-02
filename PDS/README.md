# Set up Process for Stormpy in AALpy

In order to use Stormpy in AAlpy, you need to set it up in the first place which can be done in two ways. In the course of this README, you assume that the project runs on a Linux distribution, thus, you discuss the set up process for Linux (or at least in a unix shell). However, Stormpy is used in combination with Property-Directed Sampling (PDS) in AALpy and runs on an own fork of the project:
> https://github.com/MDrey101/AALpy/tree/pds

Additionally, some small changes had to be made to the original AALpy project in order to run Stormpy in the forked AALpy with Property-Directed Sampling:

- SamplingBasedObservationTable.py: n_resample > 0  
  always set resample_value (except 0)
- StochasticLStar.py: 
  - added evaluation=False in run_stochastic_Lstar method  
    has to be set to True in order to evaluate PDS
  - deleted chaos_cex_present condition  
    prevent to simply continue learning if chaos_cex_present is True
  - added execute_pds call if evaluation is True  
    call to evaluate PDS Learning
- StochasticTeacher.py: return value in pre method  
  should always be performed

## Pip package

The first possibility is to get Stormpy via pip as every other package in a project. you simply need to download and build it via pip:

> pip install Stormpy

However, at this point, the current version maintained by pip is 1.3, whereas the newest version of Stormpy is 1.6.3.  
If someone wants the newer version of Stormpy, the other possibility to set up Stormpy has to be performed.

## Setting up Stormpy

According to the [Stormpy site](https://moves-rwth.github.io/Stormpy/) the requirements for Stormpy are

- Python3
- [Pycarl]{https://moves-rwth.github.io/Pycarl/index.html}
- [Storm]{https://www.Stormchecker.org/index.html}

For every dependency of Stormpy, a virtual environment can be created to better maintain their own dependencies.

> pip install virtualenv  
> virtualenv -p python3 env  
> source env/bin/activate

This is one way to set up a virtual environment for python. Other ways would be using python3 directly or via pipenv.

### Pycarl

In order to install Pycarl, Carl has to be available on our system (if not already present). To achieve this, you simply have to download it via our package manager. Once this is done, Pycarl can be installed according to their website.

The first step to set up Pycarl is to clone it into a directory:
> git clone https://github.com/moves-rwth/Pycarl.git  
> cd Pycarl

After that, Pycarl can be build in development mode via:
> python3 setup.py develop  

or  

> pip install -ve  

Optional arguments can be used to customize the configuration of Pycarl. However, for our purpose the basic command is sufficient.

The last step that should be done with Pycarl is to test if everything works correctly. This can be done via:
> pip setup.py test

or

> pip install pytest  
> py.test tests/

### Storm

After Pycarl is installed, Stormpy can be installed either from source or via a package manager. In our case, you installed it from source, thus, you cover the set up process for this method here.

It is important to mention here that Storm and Stormpy are continuously extended and modified, therefore, stable releases, or at least compatible versions, should be used for both!

As you have done with Stormpy, the first step is to clone Storm in a directory:
> git clone -b stable https://github.com/moves-rwth/Storm-git

Another way would be to download a zip archive:
> wget https://github.com/moves-rwth/Storm/archive/stable.zip  

(or simply download the zip manually and enter)

> unzip stable.zip

If someone wants the most recent version, "stable" has to be replaced with "master".

The set up page of Storm suggests creating an environment variable to ease the set up process:
> export Storm_DIR=<path_to_Storm_root>

You only need the basic set up of Stormpy. If someone wants to set up a different version, Storm offers a [configuration guide]{https://www.Stormchecker.org/documentation/obtain-Storm/manual-configuration.html}.

It is suggested that a dedicated build folder is created inside the cloned Storm directory to prevent breaking something.
> cd $Storm_DIR  
> mkdir build  
> build

Inside the build directory, cmake is executed
> cmake ..

If no errors and warnings occur, you can proceed onwards to compile Storm.  
To compile everything you need for Storm, you simply have to enter
> make

If someone does not want to test Storm afterward, it is sufficient to only compile the binaries.
> make binaries

In order to run Storm from everywhere, you need to enter the following command. However, since you only need Storm to build Stormpy, it is not necessary to execute this command.
> export PATH=$PATH:$Storm_DIR/build/bin 

The last step that should be done is to test Storm if everything works correctly, thus, you can run
> make check

### Stormpy

If Pycarl and Storm work correctly, you can finally set up Stormpy for our purpose.  
As you already mentioned before, Storm and Stormpy should use compatible versions, thus the same branch should be used for Stormpy as you did with Storm.

The first step to set up Stormpy is, once again, to clone it into a directory:
> git clone https://github.com/moves_rwth/Stormpy.git  
> cd Stormpy

if the latest release was used for Storm, you have to use:
> git clone https://github.com/moves-rwth/Stormpy.git --branch 1.6.3  
> cd Stormpy

After that, Pycarl can be built in development mode via:
> python3 setup.py develop  

or  

> pip install -ve

Optional arguments can be used to customize the configuration of Pycarl. However, for our purpose the basic command is sufficient.

As you described before, it is suggested that Stormpy is tested if everything works correctly:
> python setup.py test

or

> pip install pytest  
> py.test tests/

## Import into AALpy

Once the whole set up process is finished, it can be used in AALpy. To do so, you need to set up a virtual environment for the project via the commands below or similar means.
> pip install virtualenv  
> virtualenv -p python3 env  
> source env/bin/activate

you then can proceed to clone the project into a directory:
> git clone https://github.com/DES-Lab/AALpy.git

The last step you have to do to use Stormpy in AALpy, is to import Stormpy into the virtual environment that you use with AALpy:
> pip install <path_to_Stormpy>

If this executes without errors or warnings, Stormpy is ready and can be used in AAlpy.

## AALpy

At this point, everything is finished and ready to be used. Therefore, you create a new virtual environment to be used in AALpy
> pip install virtualenv  
> virtualenv -p python3 env  
> source env/bin/activate

Now you clone the project into a directory:
> git clone https://github.com/MDrey101/AALpy.git

or you can fork the original AALpy project in github and clone the own forked repo as above (keep in mind that the changes described in the first section of the README have to be performed when forking the actual repository)

The last step is to open an Editor or IDE with the project and you are good to go.