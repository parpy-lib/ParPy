
metal_cpp_path = None

def get_metal_cpp_header_path():
    return metal_cpp_path

def set_metal_cpp_header_path(path):
    global metal_cpp_path
    print(f"setting metal c++ path to {path}")
    metal_cpp_path = path
