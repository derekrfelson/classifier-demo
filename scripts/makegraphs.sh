#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: makegraphs.sh [dot-file-directory]"
    exit 1
fi

for f in "$1"/*.dot; do
    newFile="`echo "$f" | sed -e "s/\.dot$/\.png/"`"
    echo "dot -Tpng \"$f\" > \"$newFile\""
    dot -Tpng "$f" > "$newFile"
done
