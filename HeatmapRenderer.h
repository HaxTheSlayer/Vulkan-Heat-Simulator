#pragma once
#include "VulkanContext.h"
#include "PDESolver.h"
#include <vulkan/vulkan_raii.hpp>
#include <vector>

class HeatmapRenderer {
public:
    HeatmapRenderer(VulkanContext& context, const PDESolver& solver);
    ~HeatmapRenderer() = default;

    void recordDrawCommands(vk::raii::CommandBuffer& cmdBuffer, uint32_t imageIndex, uint32_t currentFrame);

private:
    VulkanContext& vkContext;
    const PDESolver& pdeSolver;

    // Descriptors (To map the PDE Storage Buffers to the Fragment Shader)
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // The Graphics Pipeline (The compiled vertex/fragment shaders and rendering rules)
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    // --- Private Setup Helper Functions ---
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::vector<char> readFile(const std::string& filename);
};
