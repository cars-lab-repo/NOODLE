from pyverilog.vparser.parser import parse
import os
import subprocess


def parse_verilog_file(file_path):
    with open(file_path, 'r') as f:
        verilog_code = f.read()
    return parse(verilog_code)


def convert_to_dot(verilog_files_folder, output_dot_file):
    with open(output_dot_file, 'w') as dot_file:
        dot_file.write('digraph G {\n')

        for file_name in os.listdir(verilog_files_folder):
            if file_name.endswith('.v'):
                module_name = os.path.splitext(file_name)[0]
                ast, _ = parse_verilog_file(os.path.join(verilog_files_folder, file_name))
                dependencies = [dep.name for dep in ast.children() if hasattr(dep, 'name')]
                dot_file.write(f'  {module_name} [shape=box];\n')
                dot_file.write(f'  {module_name} -> {{ {"; ".join(dependencies)} }};\n')

        dot_file.write('}\n')


def process_verilog_folder(verilog_files_folder, output_dot_file):
    batch_size = 10  # Adjust this number based on your system's limitations

    verilog_files = [f for f in os.listdir(verilog_files_folder) if f.endswith('.v')]
    batches = [verilog_files[i:i + batch_size] for i in range(0, len(verilog_files), batch_size)]

    for batch in batches:
        subprocess.run(['iverilog'] + [os.path.join(verilog_files_folder, f) for f in batch])

    convert_to_dot(verilog_files_folder, output_dot_file)

# Specify the folder containing Verilog files and the output .dot file
verilog_folder = '/Users/rahvis/PycharmProjects/json/test/json'
output_dot_file = '/Users/rahvis/PycharmProjects/json/test/json/output.dot'

convert_to_dot(verilog_folder, output_dot_file)
