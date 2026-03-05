#pragma once
#include "VulkanContext.h"
#include <vulkan/vulkan_raii.hpp>
#include <vector>

struct ParameterBuffer {
    float deltaTime;
    float alpha;
    float dx;
    float dy;
    uint32_t width;
    uint32_t height;
    float mouseX;         
    float mouseY;         
    float isMouseDown;
};

class PDESolver {
public:
    PDESolver(VulkanContext& context, uint32_t gridWidth, uint32_t gridHeight);
    ~PDESolver();

    void dispatchCompute(vk::raii::CommandBuffer& cmdBuffer, float deltaTime, float mouseX, float mouseY, bool isMouseDown);
    uint32_t getCurrentFrame() const { return currentFrame; }
    const std::vector<vk::raii::Buffer>& getStorageBuffers() const { return storageBuffers; }

private:
    VulkanContext& vkContext;
    uint32_t width;
    uint32_t height;
    uint32_t currentFrame = 0; //Max frames = 2

    vk::raii::Buffer uniformBuffer = nullptr;
    vk::raii::DeviceMemory uniformMemory = nullptr;
    void* mappedUniformMemory = nullptr;

    std::vector<vk::raii::Buffer> storageBuffers; 
    std::vector<vk::raii::DeviceMemory> storageMemory;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

    void createUniformBuffer();
    void createStorageBuffers();
    void createDescriptorSets();
    void createComputePipeline();

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    static std::vector<char> readFile(const std::string& filename);
};