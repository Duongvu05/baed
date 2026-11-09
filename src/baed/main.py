from loguru import logger
from baed import hello

def main():
    logger.info(hello())

if __name__ == "__main__":
    main()
