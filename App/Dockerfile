FROM python

WORKDIR /home/app

RUN pip install boto3 pandas gunicorn streamlit scikit-learn matplotlib seaborn plotly
COPY . .

CMD streamlit run --server.port $PORT app.py