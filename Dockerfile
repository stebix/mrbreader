FROM python:3.7.10-slim-buster

RUN mkdir mrbreader/
WORKDIR mrbreader/
COPY . .

RUN pip install gdcm pydicom Pillow numpy h5py pynrrd pytest

CMD ["python", "./transduce.py", "--help"]

