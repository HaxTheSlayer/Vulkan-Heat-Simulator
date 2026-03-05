#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Binding 0: The PDE Solver's Storage Buffer
layout(std430, binding = 0) readonly buffer Grid {
    float temperatureGrid[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} pcs;

vec3 getHeatMapColor(float temp) {
    float normalizedTemp = clamp(temp / 100.0, 0.0, 1.0);
    
    vec3 coldColor = vec3(0.0, 0.0, 1.0); // Blue
    vec3 hotColor  = vec3(1.0, 0.0, 0.0); // Red
    
    return mix(coldColor, hotColor, normalizedTemp);
}

void main() {
    uint x = uint(inUV.x * pcs.width);
    uint y = uint(inUV.y * pcs.height);

    x = min(x, pcs.width - 1);
    y = min(y, pcs.height - 1);

    // 2. Flatten the 2D coordinate into the 1D array index
    uint index = y * pcs.width + x;

    float temp = temperatureGrid[index];
    outColor = vec4(getHeatMapColor(temp), 1.0);
}