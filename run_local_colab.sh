pip install -q --upgrade jupyter_http_over_ws 

jupyter serverextension enable --py jupyter_http_over_ws

cd ..

jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --no-browser
