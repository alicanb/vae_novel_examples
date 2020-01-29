import argparse
import os

from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

parser = argparse.ArgumentParser(description='Download necessary models')
parser.add_argument('--root', type=str, default='./models/',
                    help='models will be downloaded in this folder')

args = parser.parse_args()

MODELS_ID = "1FnfaIv7J28ek77U4XwQv993McxBUxzLu"


def download_and_extract_archive(file_id, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root

    download_file_from_google_drive(file_id, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


if __name__ == "__main__":
    download_and_extract_archive(MODELS_ID, args.root, filename='models.zip', remove_finished=True)
