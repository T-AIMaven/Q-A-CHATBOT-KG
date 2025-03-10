import pandas as pd
import os
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVSource:
    def __init__(self, directory=settings._file_path):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
        self.current_file_index = 0
        self.df = None
        self.index = 0
        self.load_next_file()

    def load_next_file(self):
        if self.current_file_index < len(self.files):
            file_path = self.files[self.current_file_index]
            try:
                self.df = pd.read_csv(file_path, encoding="latin1")
                self.index = 0  # Initialize an index to track the current row
                logger.info(f"Successfully loaded CSV file from {file_path}")
                self.current_file_index += 1
            except UnicodeDecodeError:
                logger.warning(f"Failed to read CSV file with encoding, trying next encoding.")
                self.current_file_index += 1
                self.load_next_file()
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                self.current_file_index += 1
                self.load_next_file()
            except pd.errors.EmptyDataError:
                logger.error(f"Empty data: {file_path}")
                self.current_file_index += 1
                self.load_next_file()
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                self.current_file_index += 1
                self.load_next_file()
        else:
            self.df = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.df is not None and self.index < len(self.df):
            row = self.df.iloc[self.index]  # Get the current row
            self.index += 1  # Move to the next row
            return row.to_dict()  # Return the row as a dictionary
        else:
            self.load_next_file()
            if self.df is not None:
                return self.__next__()
            else:
                raise StopIteration  # Stop iteration when all rows are processed

csv_source = CSVSource()