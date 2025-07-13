import os

metal_cpp_path = os.getenv("METAL_CPP_HEADER_PATH")

def get_metal_cpp_header_path():
    return metal_cpp_path

def set_metal_cpp_header_path(path):
    global metal_cpp_path
    metal_cpp_path = path
