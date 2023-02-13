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
    
    
    def download_cleaned(self, fp_dest: Path, ident=None) -> None:
        """ Download from mega
        dest:  Path
            Path to the cleaned file
        """
        fn_cleaned = self.fn_cleaned[ident] if ident is not None else self.fn_cleaned
        url_cleaned = self.url_cleaned[ident] if ident is not None else self.url_cleaned

        self._download_from_mega(fp_dest.parent, fn_cleaned, url_cleaned, unzip=False)
        import shutil
        src = fp_dest.parent.joinpath(fn_cleaned)
        sink = fp_dest
        shutil.move(src, sink)


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

