#include "PDESolver.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

PDESolver::PDESolver(VulkanContext& context, uint32_t gridWidth, uint32_t gridHeight)
    : vkContext(context), width(gridWidth), height(gridHeight)
{
    createUniformBuffer();
    createStorageBuffers();
    createDescriptorSets();
    createComputePipeline();
}

PDESolver::~PDESolver() {
    if (mappedUniformMemory && *uniformMemory) {
        uniformMemory.unmapMemory();
    }
}

void PDESolver::createUniformBuffer() {

    vk::DeviceSize bufferSize = sizeof(ParameterBuffer);
    vk::BufferCreateInfo bufferInfo(
        {},                                       // flags
        bufferSize,                               // size
        vk::BufferUsageFlagBits::eUniformBuffer,  // usage
        vk::SharingMode::eExclusive               // sharingMode
    );
    uniformBuffer = vk::raii::Buffer(vkContext.getDevice(), bufferInfo);

    vk::MemoryRequirements memRequirements = uniformBuffer.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo(
        memRequirements.size,                     // allocationSize
        findMemoryType(                           // memoryTypeIndex
            memRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        )
    );
    uniformMemory = vk::raii::DeviceMemory(vkContext.getDevice(), allocInfo);

    uniformBuffer.bindMemory(*uniformMemory, 0);
    mappedUniformMemory = uniformMemory.mapMemory(0, bufferSize);
}

void PDESolver::createStorageBuffers() {
    vk::DeviceSize SbufferSize = sizeof(float) * width * height;

    for (int i = 0; i < 2; i++)
    {
        vk::BufferCreateInfo SbufferInfo(
            {},                                                                                 // flags
            SbufferSize,                                                                        // size
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,    // usage
            vk::SharingMode::eExclusive                                                         // sharingMode
        );
        vk::raii::Buffer buffer(vkContext.getDevice(), SbufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo(
            memRequirements.size,
            findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent 
            )
        );
        vk::raii::DeviceMemory memory(vkContext.getDevice(), allocInfo);

        buffer.bindMemory(*memory, 0);
        storageBuffers.emplace_back(std::move(buffer));
        storageMemory.emplace_back(std::move(memory));
    }

    std::vector<float> initialTemp(width * height, 0.0f);

    // Create a 20x20 block of intense heat in the center of the grid
    for (int y = height / 2 - 10; y < height / 2 + 10; y++) {
        for (int x = width / 2 - 10; x < width / 2 + 10; x++) {
            initialTemp[y * width + x] = 5000.0f;
        }
    }

    // Map Buffer 0 to the CPU, copy the heat data in, and unmap it
    void* mappedData = storageMemory[0].mapMemory(0, SbufferSize);
    memcpy(mappedData, initialTemp.data(), (size_t)SbufferSize);
    storageMemory[0].unmapMemory();
}

void PDESolver::createDescriptorSets() {
    std::array layoutBindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr) };

    vk::DescriptorSetLayoutCreateInfo layoutInfo(
        {},                                                 //flags
        static_cast<uint32_t>(layoutBindings.size()),       //bindingCount
        layoutBindings.data()                               //pBindings
    );
    descriptorSetLayout = vk::raii::DescriptorSetLayout(vkContext.getDevice(), layoutInfo);

    std::array poolSize{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4) };
    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,       //flags
        2,                                                          //maxSets
        poolSize.size(),                                            //poolSizeCount
        poolSize.data()                                             //pPoolSizes
    );
    descriptorPool = vk::raii::DescriptorPool(vkContext.getDevice(), poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(2, *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptorPool,        // descriptorPool
        2,                      // descriptorSetCount
        layouts.data()          // pSetLayouts (Pointer to the array!)
    );

    descriptorSets.clear();
    descriptorSets = vkContext.getDevice().allocateDescriptorSets(allocInfo);

    for (int i = 0; i < 2; i++)
    {
        vk::DescriptorBufferInfo uniformInfo(
            *uniformBuffer,              //buffer
            0,                          //offset
            sizeof(ParameterBuffer)     //range
        );

        int readIndex = i;
        int writeIndex = (i + 1) % 2;
        
        vk::DescriptorBufferInfo storageReadInfo(
            *storageBuffers[readIndex],     //buffer
            0,                              //offset
            VK_WHOLE_SIZE                   //range
        );

        vk::DescriptorBufferInfo storageWriteInfo(
            *storageBuffers[writeIndex],    //buffer
            0,                              //offset
            VK_WHOLE_SIZE                   //range
        );

        std::array descriptorWrites{
            vk::WriteDescriptorSet(
                *descriptorSets[i],                     //dstSet
                0,                                      //dstBinding
                0,                                      //dstArrayElement
                1,                                      //descriptorCount
                vk::DescriptorType::eUniformBuffer,     //descriptorType
                nullptr,                                //pImageInfo
                &uniformInfo,                            //pBufferInfo
                nullptr                                 //pTexelBufferView
            ),
            vk::WriteDescriptorSet(
                *descriptorSets[i],                     //dstSet
                1,                                      //dstBinding
                0,                                      //dstArrayElement
                1,                                      //descriptorCount
                vk::DescriptorType::eStorageBuffer,     //descriptorType
                nullptr,                                //pImageInfo
                &storageReadInfo,                       //pBufferInfo
                nullptr                                 //pTexelBufferView
            ),
            vk::WriteDescriptorSet(
                *descriptorSets[i],                     //dstSet
                2,                                      //dstBinding
                0,                                      //dstArrayElement
                1,                                      //descriptorCount
                vk::DescriptorType::eStorageBuffer,     //descriptorType
                nullptr,                                //pImageInfo
                &storageWriteInfo,                       //pBufferInfo
                nullptr                                 //pTexelBufferView
            ),
        };
        vkContext.getDevice().updateDescriptorSets(descriptorWrites, {});
    }
}


void PDESolver::createComputePipeline() {
    std::vector <char> file = readFile("shaders/slang.spv");
    vk::ShaderModuleCreateInfo createInfo(
        {},                                                 //flags
        file.size(),                                        //codeSize
        reinterpret_cast<const uint32_t*>(file.data())      //pcode
    );
    vk::raii::ShaderModule shaderModule{vkContext.getDevice(), createInfo };

    vk::PipelineShaderStageCreateInfo computeShaderStageInfo(
        {},                                     //flag
        vk::ShaderStageFlagBits::eCompute,      //stage
        shaderModule,                           //module
        "main"                                  //pName
    );

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
        {},                         //flags
        1,                          //setlayerCount
        &*descriptorSetLayout       //pSetLayout
    );
    pipelineLayout = vk::raii::PipelineLayout(vkContext.getDevice(), pipelineLayoutInfo);

    vk::ComputePipelineCreateInfo pipelineInfo(
        {},                         //flags
        computeShaderStageInfo,     //stage
        *pipelineLayout             //layout
    );
    computePipeline = vk::raii::Pipeline(vkContext.getDevice(), nullptr, pipelineInfo);

}

void PDESolver::dispatchCompute(vk::raii::CommandBuffer& cmdBuffer, float deltaTime, float mouseX, float mouseY, bool isMouseDown) {
    ParameterBuffer params{};
    params.deltaTime = 0.016f;
    params.alpha = 15.5f;
    params.dx = 1.0f;
    params.dy = 1.0f;

    params.width = width;
    params.height = height;

    params.mouseX = mouseX;
    params.mouseY = mouseY;
    params.isMouseDown = isMouseDown ? 1.0f : 0.0f;

    memcpy(mappedUniformMemory, &params, sizeof(ParameterBuffer));

    auto& commandBuffer = cmdBuffer;
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, { descriptorSets[currentFrame] }, {});
    commandBuffer.dispatch(width / 16, height / 16, 1);

    currentFrame = (currentFrame + 1) % 2;
}

// --- Helper Function ---
// (You can copy the exact implementation of this from the Sascha Willems code!)
uint32_t PDESolver::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = vkContext.getPhysicalDevice().getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

std::vector<char> PDESolver::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
     
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    return buffer;
}
