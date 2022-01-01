import zipfile


def extract_archive(archive_path: str, target_dir: str) -> None:
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
