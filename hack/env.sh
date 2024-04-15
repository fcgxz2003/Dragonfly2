set -o nounset
set -o errexit
set -o pipefail

export INSTALL_HOME=/opt/dragonfly
export INSTALL_BIN_PATH=bin
export GO_SOURCE_EXCLUDES=( \
    "test" \
)

export LIBRARY_PATH=/usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib

GOOS=$(go env GOOS)
GOARCH=$(go env GOARCH)
CGO_ENABLED=${CGO_ENABLED:-0}
export GOOS
export GOARCH
export CGO_ENABLED
