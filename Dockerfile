FROM python:3.6.4-stretch

# user id for jupyter
ARG user_id=1000

RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install jupyter
RUN pip3 install pymongo

# ENV
ENV NUMBER_PROCESSES=8

# MAKE JUPYTER USER
RUN useradd -ms /bin/bash jupyter
RUN usermod -u $user_id jupyter
RUN groupmod -g $user_id jupyter

USER jupyter
WORKDIR /home/jupyter

# MAKE DEAFULT CONFIG
RUN jupyter notebook --generate-config
RUN mkdir host_data

# ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
# ENTRYPOINT ["/bin/bash"]
