
from config import Config
from loader import DataGenerator


data_path = "./text_categories_test/text_categories_test_info.csv"

def main(config):
    dg = DataGenerator(data_path, config)


if __name__ == "__main__":
    main(Config)
