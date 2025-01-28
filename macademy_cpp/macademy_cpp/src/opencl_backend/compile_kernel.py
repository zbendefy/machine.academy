import os
import sys
print("Compiling OpenCL kernel...")
print("  Arguments: {}".format(sys.argv))
os.chdir(sys.argv[1])

kernel_sources = sys.argv[2:]

success = True

for kernel in kernel_sources:
    content = ""
    if not os.path.isfile(kernel):
        print("Did not find input file: {}".format(kernel))
        success = False
        break
    with open(kernel, "r") as f:
        content = 'constexpr const char* opencl_kernel_source = R"OPENCLSRC({})OPENCLSRC";'.format(f.read())
    with open(kernel + ".h", "w") as text_file:
            text_file.write(content)
    
print()
print("##################")
if success:
    print("SUCCESS")
else:
    print("FAILED")
print("################")

if not success:
    sys.exit(1)
    

