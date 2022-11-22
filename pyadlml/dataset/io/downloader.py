import zipfile
from pathlib import Path
from .local import _move_files_to_parent_folder


class DatasetDownloader():
    def __init__(self,):
        pass

class MegaDownloader(DatasetDownloader):
    def __init__(self, url, fn, url_cleaned, fn_cleaned):
        self.mega_url =  url
        self.fn = fn
        self.url_cleaned = url_cleaned
        self.fn_cleaned = fn_cleaned
    
    
    def download_cleaned(self, dest: Path) -> None:
        """ Download from mega"""
        self._download_from_mega(dest, self.fn_cleaned, self.url_cleaned, unzip=False)

    def download(self, dest: Path) -> None:
        self._download_from_mega(dest, self.fn, self.mega_url, unzip=True)

    def _download_from_mega(self, path_to_folder, file_name, url, unzip=True):
        """ Downloads dataset from MEGA and extracts it
        Parameters
        ----------
        path_to_folder : PosixPath or str
            The folder where the archive will be extracted to
        file_name : str
            The name of the file to be downloaded
        url : str
            The internet address where mega downloads the file
        unzip : bool, default=True

        """
        file_dp = Path(path_to_folder).joinpath(file_name)
        from mega import Mega

        # Download from mega
        m = Mega()    
        m.download_url(url, dest_path=str(path_to_folder), dest_filename=file_name)

        # Unzip data, remove
        if unzip:
            with zipfile.ZipFile(file_dp, "r") as zip_ref:
                zip_ref.extractall(path_to_folder)
            Path(file_dp).unlink()
            _move_files_to_parent_folder(path_to_folder.joinpath(file_name[:-4]))

