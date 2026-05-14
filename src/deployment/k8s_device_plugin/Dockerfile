FROM golang:1.22-alpine AS builder
WORKDIR /build
COPY main.go go.mod ./
RUN go mod tidy && go build -o vgre-device-plugin main.go

FROM alpine:3.19
COPY --from=builder /build/vgre-device-plugin /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/vgre-device-plugin"]
