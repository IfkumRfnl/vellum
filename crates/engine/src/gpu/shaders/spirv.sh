#!/bin/sh
# Compiles the GLSL kernels (`*.comp`) to the SPIR-V the engine embeds (`spv/*.spv`).
# WGSL can't name int8 cooperative matrices, so those kernels are GLSL
# (GL_KHR_cooperative_matrix); the SPIR-V is checked in, so building the engine needs no
# Vulkan SDK. Run after changing a `.comp`, and commit the `.spv` with it.
#
# The checked-in files are exactly what this script writes with glslc from the Vulkan SDK
# 1.4.350.0 (`glslc --version`: shaderc v2026.2, spirv-tools v2026.2, glslang
# 11.1.0-1412-g5ed4003a); `spirv-val --target-env vulkan1.3` accepts them. Another
# compiler version may write different (equally valid) bytes: check the result with
# `cargo lab kernels --kernels q8`, which runs the multiply against the CPU.
set -e
cd "$(dirname "$0")"
for f in *.comp; do
    glslc --target-env=vulkan1.3 -O "$f" -o "spv/${f%.comp}.spv"
    spirv-val --target-env vulkan1.3 "spv/${f%.comp}.spv"
done
