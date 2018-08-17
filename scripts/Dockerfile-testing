FROM xgboost:latest

# Install golang
RUN apt-get update
RUN apt-get install -y wget pkg-config git gcc

RUN wget -P /tmp https://storage.googleapis.com/golang/go1.10.2.linux-amd64.tar.gz

RUN tar -C /usr/local -xzf /tmp/go1.10.2.linux-amd64.tar.gz
RUN rm /tmp/go1.10.2.linux-amd64.tar.gz

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

