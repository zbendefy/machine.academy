from cgi import test
from genericpath import isfile
import os
import sys
import base64

print("Compiling shaders")

shader_folder = "."
output_folder = "."
config = "debug"
macros = []
shader_sources = [
    "kernel_calc_single_layer.glsl",
    "kernel_training_forward_pass.glsl",
    "kernel_training_backward_pass.glsl",
    "kernel_apply_gradient.glsl"
                  ]

VulkanSDKFolder = os.environ['VULKAN_SDK']
print("Vulkan SDK folder: '{}'".format(VulkanSDKFolder))

def FindVulkanSDKToolPath(binary_name):
    possible_paths = [
        "/Bin/{}.exe".format(binary_name), # windows vulkansdk path
        "/bin/{}".format(binary_name),
        "/x86_64/bin/{}".format(binary_name), # shouldn't be needed
        "/Bin/{}".format(binary_name) # shouldn't be needed
    ]
    ret = ""
    for test_path in possible_paths:
        full_test_path = "{}{}".format(VulkanSDKFolder, test_path)
        if os.path.isfile(full_test_path):
            ret = full_test_path
            print("Found {}: {}".format(binary_name, ret))
    if ret == "":
        print("Did not find {}!".format(binary_name))
        exit()
    return ret;

    
glslc_path = FindVulkanSDKToolPath("glslc")

def CompileVulkanShader(shader_folder, shader_filename, glslc_args):
    shader_full_path = shader_folder + "/" + shader_filename
    #Compile Vulkan shaders    
    print(" * Compiling {} --> {}.spv".format(shader_filename, shader_filename))

    result = os.system("{} -fshader-stage=comp {} -o {}.spv {}".format(glslc_path, shader_full_path, shader_filename, glslc_args))
    if result == 0:
        base64_content = ""
        with open(shader_filename + ".spv", "rb") as binary_file:
            binary_data = binary_file.read()
            encoded_data = base64.b64encode(binary_data)
            base64_content = encoded_data.decode("utf-8")
        with open(shader_filename + ".h", "w") as text_file:
            name = shader_filename.replace(".", "_")
            text_file.write('constexpr const char* vulkan_kernel_source_{}_b64 = '.format(name))
            while len(base64_content) > 0:
                fragment_size = min(len(base64_content), 80)
                fragment = base64_content[:fragment_size]
                base64_content = base64_content[fragment_size:]
                text_file.write('"{}"\n'.format(fragment))
            text_file.write(";")

    return result == 0

success = True

print("Compiling vulkan shaders")
glslc_args = "-O0" if config.lower() == "debug" else "-O"
for m in macros:
    glslc_args = glslc_args + " {}".format(m)
for shader in shader_sources:
    vk_result = CompileVulkanShader(shader_folder, shader, glslc_args)
    if vk_result == False:
        success = False
        
print()

print()
print("##################")

if success:
    print("SUCCESS")
else:
    print("FAILED")
 
print("################")

if not success:
    sys.exit(1)