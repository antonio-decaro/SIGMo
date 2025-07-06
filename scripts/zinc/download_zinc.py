import argparse
import gdown

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download ZINC dataset from Google Drive")
  parser.add_argument(
    "-o", "--output", type=str, required=True,
    help="Output filename (e.g., zinc.zst)"
  )
  args = parser.parse_args()

  # Google Drive file ID
  file_id = "1L02GgTgVl8KZQRTygIkhePputyHikwdO"

  # Construct the download URL
  url = f"https://drive.google.com/uc?id={file_id}"

  # Download the file with progress bar
  gdown.download(url, args.output, quiet=False)