import pandas as pd
import electrochem as echem


def parse_arbinres(file_path: str, save_path: str) -> pd.DataFrame:
    echem.parseArbin(file_path, save_path)
    df = pd.read_csv(save_path)
    return df