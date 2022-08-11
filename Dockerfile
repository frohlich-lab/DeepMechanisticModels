FROM dweindl/amici:latest
LABEL description="MechanismEncoder"

# Clone the git repository. Get access to SSH keys without leaving them in the generated image
RUN apt-get install -y git openssh-client

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:FFroehlich/mechanismEncoder.git mechanismEncoder

# Setup the Python environment.
RUN pip3 install -r mechanismEncoder/requirements.txt
