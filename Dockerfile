FROM python:3.10


WORKDIR /projetofinal

RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc python3-dev libssl-dev

RUN pip install --upgrade pip==24.0
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN  pipenv install --system --deploy

ADD projetofinal/ projetofinal/
ADD tests/ tests/
ADD predict.py ./

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]


#CMD ["python", "-m", "projetofinal", "train"]
