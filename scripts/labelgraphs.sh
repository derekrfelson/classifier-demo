#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: makegraphs.sh [dot-file-directory]"
    exit 1
fi

for f in "$1"/Iris*.dot; do
    newFile="`echo "$f" | sed -e "s/\.dot$/\.png/"`"
    sed -f "iris_labels.sed" "$f" | dot -Tpng > "$newFile"
done

for f in "$1"/Wine*.dot; do
    newFile="`echo "$f" | sed -e "s/\.dot$/\.png/"`"
    sed -f "wine_labels.sed" "$f" | dot -Tpng > "$newFile"
done

for f in "$1"/Heart*.dot; do
    newFile="`echo "$f" | sed -e "s/\.dot$/\.png/"`"
    sed -f "heart_disease_labels.sed" "$f" | dot -Tpng > "$newFile"
done
