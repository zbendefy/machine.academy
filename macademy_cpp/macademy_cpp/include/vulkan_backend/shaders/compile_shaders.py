import os
import sys
import struct

print("Compiling Vulkan shaders...")
print("  Arguments: {}".format(sys.argv))

os.chdir(sys.argv[1])
glslc_path = sys.argv[2]
shader_sources = sys.argv[3:]

config = "Release"
macros = []

VulkanSDKFolder = os.environ['VULKAN_SDK']
print("Vulkan SDK folder: '{}'".format(VulkanSDKFolder))

def CompileVulkanShader(shader_filename, glslc_args):
    #Compile Vulkan shaders    
    print(" * Compiling {} --> {}.spv".format(shader_filename, shader_filename))

    result = os.system("{} --target-env=vulkan1.1 -fshader-stage=comp {} -o {}.spv {}".format(glslc_path, shader_filename, shader_filename, glslc_args))
    if result == 0:
        uint32_list = []
        with open(shader_filename + ".spv", "rb") as binary_file:
            while True:
                chunk = binary_file.read(4)
                if len(chunk) != 4:
                    break
                uint32_value = struct.unpack('<I', chunk)[0]
                uint32_list.append(uint32_value)
        with open(shader_filename + ".h", "w") as text_file:
            name = os.path.basename(shader_filename).replace(".", "_")
            text_file.write('std::array<uint32_t, {}> vulkan_kernel_source_{} = {{'.format(len(uint32_list), name))
            while len(uint32_list) > 0:
                fragment_size = min(len(uint32_list), 40)
                fragment = ', '.join(map(hex, uint32_list[:fragment_size]))
                uint32_list = uint32_list[fragment_size:]
                text_file.write('{},\n'.format(fragment))
            text_file.write("};")
    return result == 0

success = True
print("  Config: {}".format(config))
glslc_args = "-O0" if config.lower() == "debug" else "-O"
for m in macros:
    glslc_args = glslc_args + " {}".format(m)
    
for shader in shader_sources:
    vk_result = CompileVulkanShader(shader, glslc_args)
    if vk_result == False:
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