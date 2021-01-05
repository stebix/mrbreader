import io
import pathlib
import zipfile as zpf
import numpy as np

from typing import Union, Dict, List
from PIL import Image

PathLike = Union[str, pathlib.Path]




class AbstractMRBFile:

    def __init__(self):
        pass



class MRBFile(AbstractMRBFile):
    """
    Encapsulates/exposes access to the MRB file via standard unpacking and
    direct read access via a context manager.

    Parameters
    ----------

    filepath : PathLike
        The path pointing to the MRB file.

    """

    def __init__(self, filepath: PathLike):
        if not isinstance(filepath, pathlib.Path):
            filepath = pathlib.Path(filepath)
        assert filepath.is_file(), f'MRB file at {filepath.resolve()} not found!'

        self.filepath = filepath
        

    
    def extract(self, target_dir: PathLike):
        """
        Decompress medical reality bundel file (MRB) into given directory.

        Parameters
        ----------

        target_dir : PathLike
            The target directory. Will be created if not existing.
        """
        if not isinstance(target_dir, pathlib.Path):
            target_dir = pathlib.Path(target_dir)


    def __enter__(self):
        pass


    def __exit__(self):
        pass
        




class MRBReader:
    
    def __init__(self, filepath: PathLike):
        self.filepath = filepath
        self.file = None

    def print_content(self):
        pass


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    testfile_path = pathlib.Path('C:/Users/Jannik/Desktop/mrbreader/tests/assets/mu_ct_seg.mrb')
    assert testfile_path.is_file()

    with zpf.ZipFile(testfile_path, mode='r') as f:
        print(f.namelist())

        for item in f.infolist():
            print(item.filename)
            if item.filename.endswith('.png'):

                # img_data = Image.frombytes(f.read(item))

                img_data = Image.open(io.BytesIO(f.read(item)))

                img_data = np.asarray(img_data)


                print(f'IMG DATA SHAPE: {img_data.shape}')

                fig, ax = plt.subplots()
                ax.imshow(img_data)

                plt.show()


