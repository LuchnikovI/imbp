#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

log() {
    local severity=$1
    shift

    local ts=$(date "+%Y-%m-%d %H:%M:%S%z")

    # See https://stackoverflow.com/a/29040711 and https://unix.stackexchange.com/a/134219
    local module=$(caller | awk '
        function basename(file, a, n) {
            n = split(file, a, "/")
            return a[n]
        }
        { printf("%s:%s\n", basename($2), $1) }')

    case "${severity}" in
        ERROR)
            color_start='\033[0;31m' # Red
            ;;
        WARNING)
            color_start='\033[1;33m' # Yellow
            ;;
        INFO)
            color_start='\033[1;32m' # Light Green
            ;;
        DEBUG)
            color_start='\033[0;34m' # Blue
            ;;
    esac
    color_end='\033[0m'

    printf "# ${ts} ${color_start}${severity}${color_end} [${module}]: ${color_start}$*${color_end}\n" >&2
}
  
function traverse() {
    for file in "$1"/*
    do
        if [ ! -d "${file}" ] ; then
            if [[ "${file}" =~ test_.*\.jl ]] ; then
                log INFO "Running ${file} test file..."
                julia "${file}"
                if [[ $? -eq 0 ]] ; then
                    log INFO "Test ${file}: OK"
                else
                    log ERROR "Test ${file}: FAILED"
                fi
            fi
        else
            traverse "${file}"
        fi
    done
}

traverse "${script_dir}"
