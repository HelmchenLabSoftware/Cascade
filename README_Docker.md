## How to run Cascade in a Docker container

1. Make sure you have [Docker](https://www.docker.com/) installed
2. Download / clone the repository to your local computer
3. Set working directory of your shell / terminal to the repository
4. Build the container: `docker build -t cascade-notebook .` (this will take a while)
5. Run the container: `docker run -p 8888:8888 -v "$PWD":/home/jovyan/work cascade-notebook`
6. Open the Jupyter notebook server by pasting the URL in the output of the previous command into your browser, e.g. http://127.0.0.1:8888/?token=dfa6fcaf7c53c104dfdfdeffaaf3e487b355edb
The repository folder is mounted in the container and changes are persistent.
