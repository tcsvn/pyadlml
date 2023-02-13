import hashlib
import shutil
import pandas as pd
from pathlib import Path


def clear_data_home():
    """ Delete all content inside the data home folder.
    """
    data_home = get_data_home()
    _delete_data(data_home)
    _create_folder(data_home)

def _move_files_to_parent_folder(path_to_folder):
    """ Moves all files in given folder on level up and deletes the empty directory
    """
    import shutil
    source_dir = Path(path_to_folder)
    for file_name in source_dir.iterdir():
        shutil.move(str(source_dir.joinpath(file_name)), source_dir.parent)

    source_dir.rmdir()

def set_data_home(path_to_folder):
    """
    Sets the global variable ``data_home`` and creates the according folder.
    All dump, load and fetch operations assume this folder as the base of operation.

    Parameters
    ----------
    path_to_folder : str
        Specifies the path where the data_home is located.
    """
    from pyadlml import DATA_HOME
    global DATA_HOME
    DATA_HOME = Path(path_to_folder)


def _ensure_dh_folder_exists():
    path_to_folder = get_data_home()
    if not path_to_folder.exists():
        _create_folder(path_to_folder)


def get_data_home() -> Path:
    """ Returns the current folder where pyadlml saves all datasets to.

    Returns
    -------
    path_to_data_home : str
    """
    global DATA_HOME
    return DATA_HOME


def load(name):
    """ Loads a python object from the data_home folder if it exists.

    Parameters
    ----------
    param_dict : dict
        A dictionary identifying the object that is to be loaded. The keys and
        values have to exactly match the objects keys and values when it was
        dumped with *dump_in_data_home*.

    Examples
    --------

    Returns
    -------
    X : pd.DataFrame
        Some observations
    y : pd.DataFrame
        Some labels
    """
    # create folder name
    folder_name = hashdict2str({'str': name})
    folder_path = Path(get_data_home()).joinpath(folder_name)

    if not folder_path.exists():
        raise EnvironmentError(f'Nothing is saved in: {folder_path}')

    res = []
    for f in folder_path.glob('*'):
        df = pd.read_parquet(f)
        res.append(df)
   
    return res


def dump(dfs: list, name: str) -> None:
    """ Creates a folder inside the *data_home* and dumps the given dataframe(s) inside.

    Parameters
    ----------
    dfs : list or pd.DataFrame
        The dataframe or dataframes to save
    name : str
        A string used to generate the folder name where the dataframes are stored.

    """
    dfs = dfs if isinstance(dfs, list) else [dfs]

    _ensure_dh_folder_exists()

    # create string representation of dictionary
    folder_name = hashdict2str({'str': name})
    folder_path = Path(get_data_home()).joinpath(folder_name)

    # remove folder if it already exists
    if folder_path.exists():
        shutil.rmtree(folder_path)
        
    _create_folder(folder_path)

    # save all data
    for i, df in enumerate(dfs):
        fp = folder_path.joinpath(f'df_{i}.parquet')
        #df.to_parquet(fp)
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        pq.write_table(table, fp)


def _create_folder(path_to_folder):
    Path(path_to_folder).mkdir(parents=True, exist_ok=True)


def hashdict2str(param_dict):
    """ creates a unique string for a dictionary 
    Parameters
    ----------
    param_dict : dict
        Contains the attributes of the encoder and the data representations
    Returns
    -------
    folder_name : str
    """
    # sort dictionary after keys for determinism
    param_dict = {k: v for k, v in sorted(param_dict.items(), key=lambda item: item[1])}

    # create string
    param_string = str(param_dict).encode('utf-8')
    folder_name = hashlib.md5(param_string).hexdigest()
    return str(folder_name)


def _delete_data(path_to_folder):
    # make sure only data in home directory are deleted
    #assert '/home/' == path_to_folder[:6]
    shutil.rmtree(path_to_folder)


def _data_2_folder_name(path_to_folder, data_name):
    param_dict = {'dataset': data_name}
    folder_name = hashdict2str(param_dict)
    folder_path = Path(path_to_folder).joinpath(folder_name)
    return folder_path