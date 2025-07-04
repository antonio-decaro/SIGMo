#!/bin/bash

# take as input a file with a list of urls
# and download each line of the file in parallel with progress shown

if [ $# -ne 2 ]; then
  echo "Usage: $0 <url_file> <output_folder>"
  exit 1
fi

URL_FILE=$1
OUT_FOLDER=$2
if [ ! -f "$URL_FILE" ]; then
  echo "File not found: $URL_FILE"
  exit 1
fi
while IFS= read -r url; do
  if [[ ! -z "$url" ]]; then
    echo "Downloading $url..."
    wget --show-progress "$url" -O "$OUT_FOLDER/$(basename "$url")"
  fi
done < "$URL_FILE"
wait
echo "All downloads completed."
