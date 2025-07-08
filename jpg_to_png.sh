#!/bin/bash

# Check if directory is passed
if [ -z "$1" ]; then
    echo "Usage: $0 path/to/subdirectory"
    exit 1
fi

DIR="$1"

# Check if ImageMagick's convert is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick 'convert' command not found. Please install it."
    exit 1
fi

# Convert each .jpg/.jpeg file to .png
find "$DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | while read img; do
    out="${img%.*}.png"
    echo "Converting: $img → $out"
    convert "$img" "$out"
done

