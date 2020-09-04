#!/bin/bash

if [ $# -ne 1 ]; then
    echo "args should be equal to one"
    exit 1
fi

prefix=`date '+%Y-%m-%d'`

touch ./_posts/$prefix-$1.md