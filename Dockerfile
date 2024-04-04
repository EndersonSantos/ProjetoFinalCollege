FROM python:3.12.1


WORKDIR /projetofinal

RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc python3-dev libssl-dev

RUN pip install --upgrade pip==24.0

COPY requirements.txt ./

RUN  pip install -r requirements.txt


ADD projetofinal/ projetofinal/
ADD tests/ tests/


CMD ["python", "-m", "projetofinal", "train"]
