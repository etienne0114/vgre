FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git

WORKDIR /build

# Copy dependency manifests first for better layer caching.
# go.sum must exist in the repository; run 'go mod tidy' locally to generate it.
COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o vgre-device-plugin main.go

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /build/vgre-device-plugin /usr/local/bin/vgre-device-plugin
ENTRYPOINT ["/usr/local/bin/vgre-device-plugin"]
