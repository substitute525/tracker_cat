import argparse

parser = argparse.ArgumentParser(description="命令行参数示例")
parser_args = None

def get_parser():
    return parser

def add_argument(*args, **kwargs):
    parser.add_argument(*args, **kwargs)

def get_args():
    global parser_args
    if parser_args is None:
        parser_args = parser.parse_args()
    return parser_args
