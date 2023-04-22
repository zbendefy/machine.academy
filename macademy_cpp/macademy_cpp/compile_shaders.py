from cgi import test
from genericpath import isfile
import os
import sys

print("Compiling shaders")

if(len(sys.argv) < 3):
    print("Not enough arguments! Quitting")
    quit()

shader_folder = None
output_folder = None
config = "debug"
macros = []

for arg in sys.argv:
    arg_pair = arg.split("=")
    if len(arg_pair) == 2:
        if arg_pair[0] == "--shader_folder":
            shader_folder = arg_pair[1]
        elif arg_pair[0] == "--output_folder":
            output_folder = arg_pair[1]
        elif arg_pair[0] == "--config":
            config = arg_pair[1]
        elif arg_pair[0] == "--define":
            macros.append(arg[len("--define=")]) #substring

if shader_folder == None or output_folder == None:
    print("Missing shader folder, or output folder!")
    sys.exit(1)

possible_glslc_paths = [
    "/Bin/glslc.exe", # windows vulkansdk path
    "/bin/glslc",
    "/x86_64/bin/glslc", # shouldn't be needed
    "/Bin/glslc" # shouldn't be needed
]

VulkanSDKFolder = os.environ['VULKAN_SDK']
print("Vulkan SDK folder: '{}'".format(VulkanSDKFolder))

glslc_path = ""

for test_path in possible_glslc_paths:
    full_test_path = "{}{}".format(VulkanSDKFolder, test_path)
    if os.path.isfile(full_test_path):
        glslc_path = full_test_path
        print("Found glslc: " + glslc_path)
        break

if glslc_path == "":
    print("Did not find glslc!")
    exit()

print("Working directory is: " + os.getcwd())
shader_sources = os.listdir(shader_folder)
os.makedirs(output_folder, exist_ok=True)

args = ""

if config.lower() == "debug":
    args = "-O0"
    print("Using 'debug' profile")
else:
    args = "-O"
    print("Using 'release' profile")

for m in macros:
    args = args + " {}".format(m)

success = True
print()

for shader in shader_sources:
    if shader.endswith(".glsl"):
        continue # skip included shaders
    print(" * Compiling {}/{} --> {}/{}.spv".format(shader_folder, shader, output_folder, shader))

    result = os.system("{} {}/{} -o {}/{}.spv {}".format(glslc_path, shader_folder, shader, output_folder, shader, args))
    if result != 0:
        print() # Print a new line for easier separation
        success = False
    
print()
print("##################")

if success:
    print("SUCCESS")
else:
    print("FAILED")
 
print("################")

if not success:
    sys.exit(1)