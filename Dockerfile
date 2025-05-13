FROM python:3.13
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user ./requirements.txt ./
RUN pip install -r requirements.txt
COPY --chown=user . .
CMD ["chainlit", "run", "chainlit_app.py", "--port", "7860"]
