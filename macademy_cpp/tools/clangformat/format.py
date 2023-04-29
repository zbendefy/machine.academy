from genericpath import isfile
import os
import sys
import platform
import subprocess

print("Formatting projects")
print("Working directory: {}".format(os.getcwd()))

clang_format_path = ""
clang_format_config_path = ".clang-format"
folders = []
verbose = False
dry_run = False

for arg in sys.argv:
    arg_pair = arg.split("=")
    if len(arg_pair) == 2:
        if arg_pair[0] == "--clang_format_path":
            clang_format_path = arg_pair[1]
        if arg_pair[0] == "--clang_format_config":
            clang_format_config_path = arg_pair[1]
        elif arg_pair[0] == "--add_folder":
            folders.append(arg_pair[1])
        elif arg_pair[0] == "--verbose":
            verbose=arg_pair[1].lower() == "true"

success = True

clang_format_relative_paths = {
    "Windows" : "/win64/clang-format.exe",
    "Linux": "/linux_x64/clang-format"
}

clang_format_path = clang_format_path + clang_format_relative_paths[platform.system()]

if not os.path.isfile(clang_format_path):
    print("Could not find clang_format at {}!".format(clang_format_path))
    os.exit(-1)
else:
    print("Found clang format at: " + clang_format_path)

for folder in folders:
    print("Formatting folder: {}".format(folder))
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if (os.path.splitext(f)[1] == '.cpp' or os.path.splitext(f)[1] == '.h')]

    if files != None and len(files) > 0:
        command = [clang_format_path, "--style=file:{}".format(clang_format_config_path), "-i"]

        for file in files:
            command.append("{}".format(file.replace("\\", "/")))
            if verbose:
                print("  " + file.replace("\\", "/"))

        if not dry_run:
            result = subprocess.run(command, cwd=os.getcwd(), stdout=subprocess.PIPE)
            if verbose:
                print(result.stdout.decode('utf-8'))
        else:
            print(" ".join(command))
            print()
