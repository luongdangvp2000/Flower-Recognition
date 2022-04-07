#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade

kaggle datasets download -d alxmamaev/flowers-recognition -p data/
unzip data/flowers-recognition.zip -d data/
# mv flowers-recognition data/
# rm flowers-recognition.zip