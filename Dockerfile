FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12:latest

WORKDIR /diffusion

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src .

ENTRYPOINT ["python", "run"]
